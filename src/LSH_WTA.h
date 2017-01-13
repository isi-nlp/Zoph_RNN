//
//  LSH_WTA.h
//  lstm_github
//
//  Created by Xing Shi on 1/3/17.
//  Copyright © 2017 Xing Shi. All rights reserved.
//

#ifndef LSH_WTA_h
#define LSH_WTA_h

#include "memory_util.h"
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include "custom_kernels.h"
#include "gpu_info_struct.h"
#include "boost/range.hpp"
#include "boost/range/algorithm/sort.hpp"
#include "boost/range/algorithm_ext/push_back.hpp"
#include "boost/range/adaptor/map.hpp"
#include "thrust/fill.h"

template<typename dType>
class LSH_WTA {
public:
    int K = 8;
    int units_per_band = 2;
    int bits_per_band = 6;
    int W = 100;
    int P = 200;
    int d = 1000;
    int m = 10;
    int LSTM_size = 1000;
    int vocab_size = 1000;
    const dType NEG_FILL= -1000.0;
    // all matrix is column major
    int *d_permutes; // [K, P]
    int *h_permutes;
    //int *d_dict; //[W, vocab_size]
    unsigned int *d_bands; // [vocab_size, W]
    unsigned int *h_bands;
    dType *d_Db; // [vocab_size, LSTM_size + 1]
    Timer timer;
    bool show_debug_info = false;
    bool show_debug_info_2 = false;
    
    
    boost::random::mt19937 gen;
    boost::uniform_int<> zero_to_d;
    boost::variate_generator< boost::random::mt19937 , boost::uniform_int<> > * dice;

    // cpu version of retrival
    std::vector<std::unordered_map<unsigned int, std::vector<int>>> band_maps;
    
    LSH_WTA(int K, int units_per_band, int W, int m, int LSTM_size, int vocab_size, dType * d_D, dType * d_b, int debug_code){
        
        if (debug_code % 2 == 0){
            this->show_debug_info = false;
        } else {
            this->show_debug_info = true;
        }
        
        if ((debug_code / 2) % 2 == 0){
            show_debug_info_2 = false;
        } else {
            show_debug_info_2 = true;
        }
        
        this->m = m;
        this->K = K;
        this->units_per_band = units_per_band;
        this->bits_per_band = floor(log2(this->K*1.0)) * this->units_per_band;

        this->W = W;
        this->LSTM_size = LSTM_size;
        this->d = LSTM_size + 1;
        this->vocab_size = vocab_size;
        this->P = this->units_per_band * this->W;
        
        zero_to_d = boost::uniform_int<>(0,d-1);
        dice = new boost::variate_generator< boost::random::mt19937 , boost::uniform_int<> >(gen, zero_to_d);
        
        
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_permutes, this->K * this->P * sizeof(int)),"d_permutes failed\n");
        h_permutes = (int *) malloc(this->K * this->P * sizeof (int));
        
        // d_bands
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_bands, this->vocab_size * this->W * sizeof(unsigned int)),"d_bands failed\n");
        h_bands = (unsigned int * ) malloc (this->vocab_size * this->W * sizeof(unsigned int));
        
        // prepare d_Db
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_Db, this->vocab_size * this->d * sizeof(dType)),"d_Db failed\n");
        cudaMemcpy(d_Db,d_D,LSTM_size*vocab_size*sizeof(dType),cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_Db + LSTM_size * vocab_size, d_b, vocab_size*sizeof(dType),cudaMemcpyDeviceToDevice);
        
        // get permute
        for (int i =0 ; i < this->P; i++){
            this->get_permute(this->d, this->K, h_permutes+i*this->K);
        }
        cudaMemcpy(d_permutes, h_permutes, this->K * this->P * sizeof (int),cudaMemcpyHostToDevice);
        
        if (show_debug_info){
        std::cout <<"h_permutes\n";
        print_matrix(h_permutes, this->K, this->P);
        }
        // prepare map
        for (int i = 0; i < this-> W; i ++){
            std::unordered_map<unsigned int, std::vector<int>> map;
            band_maps.push_back(map);
        }
        
        create_hash();
    }
    
    void get_permute(int n, int k, int *data){
        std::unordered_set<int> s;
        int i = 0;
        while (i < k){
            int val = (*dice)();
            if (s.count(val) == 0 ){
                data[i] = val;
                i ++;
                s.insert(val);
            }
        }
    }
    
    
    void topm(dType *d_outputdist, dType *d_h_t, int batch_size){
        // create a new d_h_t
        dType * d_h_t_pad;
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_pad, (LSTM_size+1) * batch_size * sizeof(dType)),"d_h_t_pad failed\n");
        
        pad_h_t<<<batch_size, std::min(256, LSTM_size+1)>>>(d_h_t_pad, d_h_t, LSTM_size, batch_size);
        CUDA_GET_LAST_ERROR("pad_h_t");
        
        if (show_debug_info){
        std::cout << "d_h_t_pad\n";
        print_matrix_gpu(d_h_t_pad, batch_size, LSTM_size + 1);
        
        std::cout << "d_h_t\n";
        print_matrix_gpu(d_h_t, LSTM_size, batch_size);
        }
        

        
        //create hash_code
        
        unsigned int *d_codes;
        unsigned int *h_codes;
        
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_codes, batch_size * this->W * sizeof(unsigned int)),"d_codes failed\n");
        h_codes = (unsigned int * ) malloc(batch_size * this->W * sizeof(unsigned int));
        
        hash_code(d_codes, d_h_t_pad, batch_size);
        CUDA_GET_LAST_ERROR("hash_code in topm");
        
        if (show_debug_info){
            std::cout << "d_codes\n";
            print_matrix_gpu(d_codes, batch_size, this->W);
        }
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(h_codes, d_codes, batch_size * this->W * sizeof(unsigned int),cudaMemcpyDeviceToHost),"d_codes to h_codes");
        
        // get top_ids
        
        int *h_top_ids; // [m, batch_size]
        int *d_top_ids;
        
        allocate_matrix_dh(&h_top_ids, &d_top_ids, m, batch_size);
        
        for (int i = 0 ; i < batch_size; i ++ ){
            this->get_top_m(h_codes, h_top_ids, m, i, batch_size);
        }
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(d_top_ids, h_top_ids, m * batch_size * sizeof(int),cudaMemcpyHostToDevice),"h_top_ids to d_top_ids");
        
        if (show_debug_info_2){
            std::cout << "h_top_ids\n";
            print_matrix(h_top_ids, m, batch_size);
        }
        // init d_outputdist
        /*
        dType *d_outputdist_temp;
        dType *h_outputdist_temp;
        
        allocate_matrix_dh(&h_outputdist_temp, &d_outputdist_temp, vocab_size, batch_size);
        
        thrust::device_ptr<dType> thrust_outputdist_temp = thrust::device_pointer_cast(d_outputdist);
         
        */
        thrust::device_ptr<dType> thrust_d_outputdist = thrust::device_pointer_cast(d_outputdist);
        
        thrust::fill(thrust_d_outputdist, thrust_d_outputdist + vocab_size * batch_size , NEG_FILL);
        CUDA_GET_LAST_ERROR("thrust::fill");

        if (show_debug_info){
            std::cout << "d_outputdist_temp\n";
            std::cout << "vocab_size " << vocab_size << "\n";
            std::cout << "batch_size " << batch_size << "\n";
            print_matrix_gpu(d_outputdist,1, 1);
        }
        
        // do sparse matrix multiplication

        
        dType *h_results;
        dType *d_results;
        allocate_matrix_dh(&h_results, &d_results, m, batch_size);
        
        sparse_dot_product<<<dim3(m,batch_size),256>>>(d_outputdist, d_results, d_Db, d_h_t_pad, d_top_ids, m, LSTM_size, batch_size, vocab_size);
        CUDA_GET_LAST_ERROR("sparse_dot_product");

        if (show_debug_info){
            std::cout<< "d_outputdist_temp\n";
            print_matrix_gpu(d_outputdist,vocab_size, batch_size);
        }

        if (show_debug_info_2){
        std::cout << "d_results\n";
        print_matrix_gpu(d_results, m, batch_size);
        }

        
        /*
        free(h_outputdist_temp);
        cudaFree(d_outputdist_temp);
        */
        
        free(h_results);
        cudaFree(d_results);
        
        free(h_top_ids);
        cudaFree(d_top_ids);
        
        free(h_codes);
        cudaFree(d_codes);
        cudaFree(d_h_t_pad);
        
        
        
    }
    
    void get_top_m(unsigned int *h_codes, int * h_top_ids, int m, int batch_index, int batch_size){
        std::unordered_map<int, int> counts;
        for (int i = 0; i < this->W; i ++ ){
            unsigned int code = h_codes[i * batch_size + batch_index];
            if (this->band_maps[i].count(code) > 0){
                std::vector<int> & ids = this->band_maps[i][code];
                for (int j=0; j<ids.size(); j++){
                    int id = ids[j];
                    if (counts.count(id) == 0){
                        counts[id] = 1;
                    } else {
                        counts[id] += 1;
                    }
                }
            }
        }
        //sort counts;
        std::vector<std::pair<int, int>> values(std::begin(counts), std::end(counts));
        boost::sort(values,
                    [](const std::pair<int,int> &x, const std::pair<int,int> &y) { return x.second > y.second; });
        int i = 0;
        for (i = 0; i< std::min(int(values.size()),m) ; i ++ ){
            const auto & item = values[i];
            h_top_ids[m * batch_index + i] = item.first;
            if (show_debug_info){
                std::cout<< item.first << " " <<  item.second << "\n";
            }
        }
        
        for (; i<m ; i ++){
            h_top_ids[m * batch_index + i] = -1;
        }
        
        if (show_debug_info){
            std::cout << "\n";
        }
        
    }
    
    
    void create_hash(){

        this->hash_code(d_bands, d_Db, vocab_size);
        
        /*
        if (show_debug_info){
            std::cout<<"d_Db\n";
            print_matrix_gpu(d_Db, vocab_size, LSTM_size + 1);
            std::cout<<"d_bands\n";
            print_matrix_gpu(d_bands, vocab_size, W);
        }
        */
        
        
        cudaMemcpy(h_bands, d_bands, this->vocab_size * this->W * sizeof(unsigned int),cudaMemcpyDeviceToHost);
        for (int i = 0 ; i < this->W; i ++){
            for (int j= 0 ; j < vocab_size ; j ++ ){
                unsigned int code = h_bands[j + i * vocab_size];
                if (band_maps[i].count(code) == 0){
                    band_maps[i][code] = std::vector<int>();
                }
                band_maps[i][code].push_back(j);
            }
        }
    }
    
    
    void hash_code(unsigned int *d_codes, dType *d_vectors, int n_vectors){
        // d_vectors : [LSTM_size, beam_size]
        // d_code: [W, beam_size]
        hash_code_kernel<<<this->W, std::min(n_vectors, 256)>>>(d_codes, d_vectors, d_permutes, this->P, this->W, this->K, this->units_per_band, bits_per_band ,n_vectors);
    }
    
    
    
    
};



#endif /* LSH_WTA_h */
