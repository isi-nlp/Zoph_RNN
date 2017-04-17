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
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>


template<typename dType>
class LSH_WTA {
public:
    
    softmax_layer<dType> * p_softmax_layer;
    
    int K = 8;
    int units_per_band = 2;
    int bits_per_band = 6;
    int W = 100;
    int P = 200;
    int d = 1000;
    int m = 10;
    int LSTM_size = 1000;
    int vocab_size = 1000;
    int batch_size = 10;
    const dType NEG_FILL= -1000.0;
    // all matrix is column major
    int *d_permutes; // [K, P]
    int *h_permutes;
    int *d_bands; // [W，vocab_size]
    int *h_bands;
    dType *d_Db; // [LSTM_size + 1，vocab_size]
    dType *d_Db_shrink; // [LSTM_size + 1, nnz]
    
    dType *d_h_t_pad; // [LSTM_size + 1, batch_size]
    int *d_h_t_pad_codes; // [W, beam_size]

    int *h_bands_index; // [vocab_size, W]
    int *d_bands_index; // [vocab_size, W]
    
    int *h_key_1; // [vocab_size, W]
    int *h_value_1; // [vocab_size, W]  record the starts
    int *h_length_1; // [vocab_size, W]  record the starts
    
    int *h_key_2; // [vocab_size, W]
    int *h_value_2; // [vocab_size, W]  record the starts
    int *h_length_2; // [vocab_size, W]  record the starts
    
    int *d_key_1; // [vocab_size, W]
    int *d_value_1; // [vocab_size, W]  record the starts
    int *d_length_1; // [vocab_size, W]  record the starts
    
    int *d_key_2; // [vocab_size, W]
    int *d_value_2; // [vocab_size, W]  record the starts
    int *d_length_2; // [vocab_size, W]  record the starts
    
    dType *d_array; //[vocab_size]
    int *d_index; //[vocab_size]
    int *d_rowIdx; //[vocab_size];
    int *h_index;
    int *h_rowIdx;
    int *d_nnz, *h_nnz;
    
    thrust::device_ptr<int> thrust_index;
    thrust::device_ptr<int> thrust_rowIdx;

    
    dType* d_outputdist_shrink; //[vocab_size, batch_size];
    
    bool show_debug_info = false;
    bool show_debug_info_2 = false;
    bool dump_file = false;
    int calltime = 0;
    int topn = 0;
    int threshold = 1;
    float fnnz;
    int nnz;
    int index_size;
    
    int debug_code;
    int target_vocab_policy;
    
    cublasHandle_t cublasHandle;
    
    boost::random::mt19937 gen;
    boost::uniform_int<> zero_to_d;
    boost::variate_generator< boost::random::mt19937 , boost::uniform_int<> > * dice;

    
    void set_to_n(int *data, int size, int n){
        for (int i = 0 ; i < size; i ++){
            data[i] = n;
        }
    }
    
    // cpu version of retrival
    std::vector<std::unordered_map<int, std::vector<int>>> band_maps;
    
    LSH_WTA(int K, int units_per_band, int W, int m, int WTA_threshold, int WTA_topn, int LSTM_size, int vocab_size, int batch_size, dType * d_D, dType * d_b, int debug_code, int target_vocab_policy, softmax_layer<dType> * p_softmax_layer){
        
        if (debug_code % 2 == 1){
            this->show_debug_info = true;
        } else {
            this->show_debug_info = false;
        }
        
        if ((debug_code >> 1) % 2 == 0){
            show_debug_info_2 = false;
        } else {
            show_debug_info_2 = true;
        }

        if ((debug_code >> 2) % 2 == 0){
            dump_file = false;
        } else {
            dump_file = true;
        }

        this->debug_code = debug_code;
        
        this->p_softmax_layer = p_softmax_layer;
        this->target_vocab_policy = target_vocab_policy;
        
        
        this->m = m;
        this->K = K;
        this->units_per_band = units_per_band;
        this->bits_per_band = floor(log2(this->K*1.0)) * this->units_per_band;

        this->W = W;
        this->LSTM_size = LSTM_size;
        this->d = LSTM_size + 1;
        this->vocab_size = vocab_size;
        this->P = this->units_per_band * this->W;
        this->batch_size = batch_size;
        this->threshold = WTA_threshold;
        this->topn = WTA_topn;
        this->index_size = this->vocab_size * 5 / 4;
        
        zero_to_d = boost::uniform_int<>(0,d-1);
        dice = new boost::variate_generator< boost::random::mt19937 , boost::uniform_int<> >(gen, zero_to_d);
        
        
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_permutes, this->K * this->P * sizeof(int)),"d_permutes failed\n");
        h_permutes = (int *) malloc(this->K * this->P * sizeof (int));
        
        // d_bands
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_bands, this->vocab_size * this->W * sizeof(int)),"d_bands failed\n");
        h_bands = (int * ) malloc (this->vocab_size * this->W * sizeof(int));

        // d_bands_index
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_bands_index, this->vocab_size * this->W * sizeof(int)),"d_bands_index failed\n");
        h_bands_index = (int * ) malloc (this->vocab_size * this->W * sizeof(int));
        
        // d_key_1
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_key_1, this->index_size * this->W * sizeof(int)),"d_key_1 failed\n");
        h_key_1 = (int * ) malloc (this->index_size * this->W * sizeof(int));

        // d_value_1
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_value_1, this->index_size * this->W * sizeof(int)),"d_value_1 failed\n");
        h_value_1 = (int * ) malloc (this->index_size * this->W * sizeof(int));

        // d_key_2
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_key_2, this->index_size * this->W * sizeof(int)),"d_key_2 failed\n");
        h_key_2 = (int * ) malloc (this->index_size * this->W * sizeof(int));

        // d_value_2
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_value_2, this->index_size * this->W * sizeof(int)),"d_value_2 failed\n");
        h_value_2 = (int * ) malloc (this->index_size * this->W * sizeof(int));

        // d_length_1
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_length_1, this->index_size * this->W * sizeof(int)),"d_value_2 failed\n");
        h_length_1 = (int * ) malloc (this->index_size * this->W * sizeof(int));
 
        // d_length_2
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_length_2, this->index_size * this->W * sizeof(int)),"d_value_2 failed\n");
        h_length_2 = (int * ) malloc (this->index_size * this->W * sizeof(int));
        
        // d_h_t_pad
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_pad, (LSTM_size+1) * batch_size * sizeof(dType)),"d_h_t_pad failed\n");

        // d_h_t_pad_codes
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_h_t_pad_codes, batch_size * this->W * sizeof(int)),"d_codes failed\n");
        
        // d_array
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_array, this->vocab_size * sizeof(dType)),"d_array failed\n");
        
        // d_index d_rowIdx
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_index, this->vocab_size * sizeof(int)),"d_index failed\n");
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_rowIdx, this->vocab_size * sizeof(int)),"d_rowIdx failed\n");
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_nnz, sizeof(int)),"d_nnz failed\n");
        
        
        h_index = (int * ) malloc (this->vocab_size  * sizeof(int));
        h_rowIdx = (int * ) malloc (this->vocab_size  * sizeof(int));
        h_nnz = (int * ) malloc (sizeof(int));
        
        for(int i = 0; i < this->topn; i++){
            h_rowIdx[i] = i;
        }
        cudaMemcpy(d_rowIdx, h_rowIdx, this->topn * sizeof (int),cudaMemcpyHostToDevice);

        // d_outputdist_shrink
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_outputdist_shrink, this->vocab_size * this->batch_size * sizeof(dType)),"d_outputdist_shrink failed\n");

        
        // prepare d_Db
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_Db, this->vocab_size * this->d * sizeof(dType)),"d_Db failed\n");
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_Db_shrink, this->vocab_size * this->d * sizeof(dType)),"d_Db_shrink failed\n");
        prepare_Db<<<this->vocab_size, 512>>>(d_Db, d_D, d_b, this->vocab_size, this->LSTM_size);
        CUDA_GET_LAST_ERROR("prepare_Db");

        if (show_debug_info){
            std::cout<<"d_D\n";
            print_matrix_gpu(d_D, vocab_size, LSTM_size );
            std::cout<<"d_b\n";
            print_matrix_gpu(d_b, vocab_size,1);
            std::cout<<"d_Db\n";
            print_matrix_gpu(d_Db, LSTM_size + 1, vocab_size);
        }
        
        // preload d_Db_shrink
        cudaMemcpy(d_Db_shrink, d_Db, this->topn * this->d * sizeof (float),cudaMemcpyDeviceToDevice);

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
            std::unordered_map<int, std::vector<int>> map;
            band_maps.push_back(map);
        }
        
        cublasCreate(&cublasHandle);

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
    
    
    void topm_cpu(dType *d_outputdist, dType *d_h_t, int batch_size){
        // create a new d_h_t
        
        pad_h_t<<<batch_size, std::min(256, LSTM_size+1)>>>(d_h_t_pad, d_h_t, LSTM_size, batch_size);
        CUDA_GET_LAST_ERROR("pad_h_t");
        
        if (show_debug_info){
        std::cout << "d_h_t_pad\n";
        print_matrix_gpu(d_h_t_pad, batch_size, LSTM_size + 1);
        
        std::cout << "d_h_t\n";
        print_matrix_gpu(d_h_t, LSTM_size, batch_size);
        }
        //create hash_code
        
        int *h_codes;
        int *d_codes;
        
        CUDA_ERROR_WRAPPER(cudaMalloc((void**)&d_codes, batch_size * this->W * sizeof(int)),"d_codes failed\n");

        h_codes = (int * ) malloc(batch_size * this->W * sizeof(int));
        
        hash_code(d_codes, d_h_t_pad, batch_size);
        CUDA_GET_LAST_ERROR("hash_code in topm");
        
        if (show_debug_info){
            std::cout << "d_codes\n";
            print_matrix_gpu(d_codes, batch_size, this->W);
        }
        
        CUDA_ERROR_WRAPPER(cudaMemcpy(h_codes, d_codes, batch_size * this->W * sizeof(int),cudaMemcpyDeviceToHost),"d_codes to h_codes");
        
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
        
    }
    
    void topm(dType *d_outputdist, dType *d_h_t, int batch_size){
        //outside base = 0.0466403 sec
        // prepare d_h_t_pad 0.0004s
        
        
#ifdef TIMER_DEBUG
        if ((debug_code >> 3) % 2 == 0) {
            p_softmax_layer->model->timer.start("pad_h_t");
#endif
            pad_h_t<<<batch_size, std::min(1024, LSTM_size+1)>>>(d_h_t_pad, d_h_t, LSTM_size, batch_size);
            CUDA_GET_LAST_ERROR("pad_h_t");
#ifdef TIMER_DEBUG
            cudaDeviceSynchronize();
            p_softmax_layer->model->timer.end("pad_h_t");
        }
#endif
        
        //create hash_code for d_h_t_pad 0.013s
#ifdef TIMER_DEBUG
        if ((debug_code >> 4) % 2 == 0) {
            p_softmax_layer->model->timer.start("hash_code");
#endif
        hash_code(d_h_t_pad_codes, d_h_t_pad, batch_size);
        CUDA_GET_LAST_ERROR("hash_code in topm");
#ifdef TIMER_DEBUG
            cudaDeviceSynchronize();
            p_softmax_layer->model->timer.end("hash_code");
        }
#endif
        
        
        // 0.001s
        cudaMemset(d_outputdist, 0, vocab_size * batch_size* sizeof(dType));
        
        
#ifdef TIMER_DEBUG

        calltime += 1;
        
        if (calltime == 2 && dump_file){
            cudaDeviceSynchronize();

            std::ofstream output("d_ht_pad_codes_input.txt");
            write_matrix_GPU(d_h_t_pad_codes,this->W,batch_size,output);
            output.close();

            output.open("d_outputdist_lookup_input.txt");
            write_matrix_GPU(d_outputdist,vocab_size,batch_size,output);
            output.close();
            
            output.open("d_key1_input.txt");
            write_matrix_GPU(this->d_key_1,index_size,this->W,output);
            output.close();

            output.open("d_value1_input.txt");
            write_matrix_GPU(this->d_value_1,index_size,this->W,output);
            output.close();
            
            output.open("d_length1_input.txt");
            write_matrix_GPU(this->d_length_1,index_size,this->W,output);
            output.close();
            
            output.open("d_key2_input.txt");
            write_matrix_GPU(this->d_key_2,index_size,this->W,output);
            output.close();
            
            output.open("d_value2_input.txt");
            write_matrix_GPU(this->d_value_2,index_size,this->W,output);
            output.close();

            output.open("d_length2_input.txt");
            write_matrix_GPU(this->d_length_2,index_size,this->W,output);
            output.close();

            output.open("d_bands_index_input.txt");
            write_matrix_GPU(d_bands_index,vocab_size,this->W,output);
            output.close();

            
        }
        
#endif
        
        //search for the top ids: d_outputdist[top_id] = 1; //
#ifdef TIMER_DEBUG
            if ((debug_code >> 5) % 2 == 0) {
                p_softmax_layer->model->timer.start("lookup");
#endif
                
                cuckoo_lookup_T<<<batch_size, std::min(1024,this->W)>>>(d_h_t_pad_codes, d_outputdist, batch_size, this->vocab_size, this->W, this->index_size, this->d_key_1, this->d_value_1, this->d_length_1, this->d_key_2, this->d_value_2, this->d_length_2, this->d_bands_index);
#ifdef TIMER_DEBUG
                CUDA_GET_LAST_ERROR("cuckoo_lookup");
                cudaDeviceSynchronize();
                p_softmax_layer->model->timer.end("lookup");
            }
#endif

#ifdef TIMER_DEBUG
        if (calltime == 2 && dump_file){
            cudaDeviceSynchronize();

            std::ofstream o_outputdist("d_outputdist_input.txt");
            write_matrix_GPU(d_outputdist,vocab_size,batch_size,o_outputdist);
            o_outputdist.close();
            
            std::ofstream o_db("d_Db_input.txt");
            write_matrix_GPU(d_Db,LSTM_size+1,vocab_size,o_db);
            o_db.close();
            
            std::ofstream o_ht_pad("d_ht_pad_input.txt");
            write_matrix_GPU(d_Db,LSTM_size+1,batch_size,o_ht_pad);
            o_ht_pad.close();
        }
#endif
        
        // dense2array
#ifdef TIMER_DEBUG
        if ((debug_code >> 6) % 2 == 0) {
            p_softmax_layer->model->timer.start("dense2array");
#endif

            this->h_nnz[0] = this->topn;
            cudaMemcpy(this->d_nnz, this->h_nnz, sizeof(int), cudaMemcpyHostToDevice);
            
            int thread_size = 256;
            dim3 threads(thread_size);
            dim3 grid((this->vocab_size+threads.x-1)/threads.x);

            dense2array<<<grid, threads>>>(d_outputdist, this->vocab_size, this->batch_size, d_rowIdx, this->topn, this->threshold, this->d_nnz); //0.02ms
            

            cudaMemcpy(h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost);
            nnz = h_nnz[0];
            std::cout<<"NNZ: "<< nnz << '\n';
            cudaMemcpy(h_rowIdx + topn, d_rowIdx + topn, (nnz - topn) * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef TIMER_DEBUG
            cudaDeviceSynchronize();
            p_softmax_layer->model->timer.end("dense2array");
        }
#endif

        
        
        // fill_new_db
#ifdef TIMER_DEBUG
        if ((debug_code >> 7) % 2 == 0) {
            p_softmax_layer->model->timer.start("fill_new_db");
#endif

        int thread_x = 256;
        int thread_y = 256/ thread_x;
        fill_new_db<<<dim3((nnz - this->topn +thread_y-1)/thread_y),dim3(thread_x,thread_y)>>>(d_Db_shrink,d_Db,d_rowIdx,this->d, nnz, this->topn); // 0.47 ms
#ifdef TIMER_DEBUG
            cudaDeviceSynchronize();
            p_softmax_layer->model->timer.end("fill_new_db");
        }
#endif

        
#ifdef TIMER_DEBUG
        if ((debug_code >> 8) % 2 == 0) {
            p_softmax_layer->model->timer.start("gemm");
#endif
            float alpha = 1.f, beta = 0.f;
            if (this->target_vocab_policy != 3){
                cublasSgemm(cublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            nnz, this->batch_size, this->d, &alpha,
                    d_Db_shrink, this->d,
                    d_h_t_pad, this->d,
                    &beta,
                    d_outputdist_shrink, nnz); //1.2 ms
            } else { // == 3
                cublasSgemm(cublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            nnz, this->batch_size, this->d, &alpha,
                            d_Db_shrink, this->d,
                            d_h_t_pad, this->d,
                            &beta,
                            d_outputdist, nnz); //1.2 ms
                
            }
#ifdef TIMER_DEBUG
            cudaDeviceSynchronize();
            p_softmax_layer->model->timer.end("gemm");
        }
#endif

        
#ifdef TIMER_DEBUG
        if ((debug_code >> 9) % 2 == 0 ) {
            p_softmax_layer->model->timer.start("array2dense");
#endif
            if (this->target_vocab_policy != 3){
                fill_number<<<dim3((this->vocab_size+1024-1)/1024,this->batch_size), 1024>>>(d_outputdist,this->vocab_size,this->batch_size,(float)-1000.0);
                array2dense<<<dim3((nnz+1024-1)/1024,this->batch_size), 1024>>>(d_outputdist,d_outputdist_shrink,d_rowIdx,this->vocab_size,nnz);
            }
#ifdef TIMER_DEBUG
            cudaDeviceSynchronize();
            p_softmax_layer->model->timer.end("array2dense");
        }
#endif

        
#ifdef TIMER_DEBUG
        p_softmax_layer->model->timer.report();
        
        if (calltime == 2 && dump_file) {
            cudaDeviceSynchronize();
            
            std::ofstream o_outputdist("d_outputdist_output.txt");
            write_matrix_GPU(d_outputdist,vocab_size,batch_size,o_outputdist);
            o_outputdist.close();
        }
        
        if (show_debug_info_2){
            cudaDeviceSynchronize();

            std::cout<<"d_h_t_pad_codes\n";
            print_matrix_gpu(d_h_t_pad_codes, this->W, batch_size);
            std::cout<<"d_outputdist\n";
            if (this->target_vocab_policy == 3){
                print_matrix_gpu(d_outputdist, nnz, batch_size);
            } else {
                print_matrix_gpu(d_outputdist, this->vocab_size, batch_size);
            }
        }
#endif
    }
    
    
    void get_top_m_cpu(int *h_codes, int * h_top_ids, int m, int batch_index, int batch_size){
        std::unordered_map<int, int> counts;
        for (int i = 0; i < this->W; i ++ ){
            int code = h_codes[i * batch_size + batch_index];
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
    
    
    void create_hash_cpu(){

        this->hash_code(d_bands, d_Db, vocab_size);
        
        /*
        if (show_debug_info){
            std::cout<<"d_Db\n";
            print_matrix_gpu(d_Db, LSTM_size + 1, vocab_size);
            std::cout<<"d_bands\n";
            print_matrix_gpu(d_bands, vocab_size, W);
        }
        */
        
        
        
        cudaMemcpy(h_bands, d_bands, this->vocab_size * this->W * sizeof(int),cudaMemcpyDeviceToHost);
        for (int i = 0 ; i < this->W; i ++){
            for (int j= 0 ; j < vocab_size ; j ++ ){
                int code = h_bands[i + j * this->W];
                if (band_maps[i].count(code) == 0){
                    band_maps[i][code] = std::vector<int>();
                }
                band_maps[i][code].push_back(j);
            }
        }
    }
    
    void create_hash(){
        // create the hash on cpu
        this->create_hash_cpu();
        
        // create the cuckoo hash
        set_to_n(this->h_key_1, this->index_size * this->W, -1);
        set_to_n(this->h_key_2, this->index_size * this->W, -1);
        set_to_n(this->h_value_1, this->index_size * this->W, -1);
        set_to_n(this->h_value_2, this->index_size * this->W, -1);
        set_to_n(this->h_length_1, this->index_size * this->W, -1);
        set_to_n(this->h_length_2, this->index_size * this->W, -1);

        if (show_debug_info_2){
            std::cout<< "h_bands before \n";
            print_matrix(this->h_bands,  this->W, this->vocab_size);
        }

        
        
        for (int i =0 ; i < this->W; i ++){
            int start = 0;
            for (auto &item : band_maps[i] ){
                int code = item.first;
                std::vector<int> &word_indexes = item.second;
                int value = start;
                int length = word_indexes.size();
                
                // add the bands and bands_index
                for (int j = 0; j < word_indexes.size(); j++){
                    h_bands[i + this->W*start] = code;
                    h_bands_index[i * this->vocab_size + start] = word_indexes[j];
                    start += 1;
                }
                
                // hash (code,value) into cuckoo
                int side = 0;
                int n_time = 0;
                while (true){
                    if (n_time > 2000){
                        std::cout<<"Index need to rehash!\n";
                    }
                    n_time += 1;
                    if (side == 0){
                        int key = (hash_func_1(code) % this->index_size + this->index_size) % this->index_size + i * this->index_size;
                        
                        if (this->h_key_1[key] == -1){
                            this->h_key_1[key] = code;
                            this->h_value_1[key] = value;
                            this->h_length_1[key] = length;
                            break;
                        } else {
                            int temp_code = this->h_key_1[key];
                            int temp_value = this->h_value_1[key];
                            int temp_length = this->h_length_1[key];
                            this->h_key_1[key] = code;
                            this->h_value_1[key] = value;
                            this->h_length_1[key] = length;
                            code = temp_code;
                            value = temp_value;
                            length = temp_length;
                            side = 1;
                        }
                    } else {
                        int key = ( hash_func_2(code) % this->index_size + this->index_size) % this->index_size + i * this->index_size;
                        if (this->h_key_2[key] == -1){
                            this->h_key_2[key] = code;
                            this->h_value_2[key] = value;
                            this->h_length_2[key] = length;
                            break;
                        } else {
                            int temp_code = this->h_key_2[key];
                            int temp_value = this->h_value_2[key];
                            int temp_length = this->h_length_2[key];
                            this->h_key_2[key] = code;
                            this->h_value_2[key] = value;
                            this->h_length_2[key] = length;
                            code = temp_code;
                            value = temp_value;
                            length = temp_length;
                            side = 0;
                        }
                    }
                }
            }
        }
        
        if (show_debug_info_2){
            std::cout<< "h_bands after \n";
            print_matrix(this->h_bands, this->W, this->vocab_size);
            std::cout<< "h_bands_index after \n";
            print_matrix(this->h_bands_index,  this->vocab_size,this->W);
            std::cout<< "h_key_1 after \n";
            print_matrix(this->h_key_1, this->index_size, this->W);
            std::cout<< "h_value_1 after \n";
            print_matrix(this->h_value_1, this->index_size, this->W);
            std::cout<< "h_key_2 after \n";
            print_matrix(this->h_key_2, this->index_size, this->W);
            std::cout<< "h_value_2 after \n";
            print_matrix(this->h_value_2, this->index_size, this->W);
            std::cout<< "band after hash_code 1 \n";
            for (int i =0 ; i< this->W; i ++ ){
            for (int j =0 ; j< this->vocab_size; j ++ ){

                    int code = this->h_bands[i*this->vocab_size + j];
                int key = (hash_func_1(code) % this->index_size + this->index_size) % this->index_size ;
                    std::cout<< key << " ";
                }
                std::cout<< "\n";
            }
            std::cout<< "band after hash_code 2 \n";
            for (int i =0 ; i< this->W; i ++ ){
                for (int j =0 ; j< this->vocab_size; j ++ ){
                    int code = this->h_bands[j*this->W + i];
                    int key = (hash_func_2(code) % this->index_size + this->index_size) % this->index_size ;
                    std::cout<< key << " ";
                }
                std::cout<< "\n";
            }
        }

        // copy to GPU
        cudaMemcpy(d_bands, h_bands, this->vocab_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_bands_index, h_bands_index, this->vocab_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_key_1, h_key_1, this->index_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_key_2, h_key_2, this->index_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_1, h_value_1, this->index_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_2, h_value_2, this->index_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_length_1, h_length_1, this->index_size * this->W * sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_length_2, h_length_2, this->index_size * this->W * sizeof(int),cudaMemcpyHostToDevice);

    }
    
    
    void hash_code(int *d_codes, dType *d_vectors, int n_vectors){
        // d_vectors : [LSTM_size, beam_size]
        // d_code: [W,beam_size]
        //hash_code_kernel<<<this->W, std::min(n_vectors, 256)>>>(d_codes, d_vectors, d_permutes, this->P, this->W, this->K, this->units_per_band, bits_per_band ,n_vectors);
        
        int n_block = this->W;
        dim3 block_dim = dim3(256,1);
        
        
        int div = 50;
        if (n_vectors < 256) {
            if (this->W < div){
                block_dim = dim3(n_vectors,1);
            } else {
                // assume W is the multiples of 50
                n_block = this->W / div;
                block_dim = dim3(n_vectors, div);
            }
        }
        
        
        
        hash_code_kernel_T<<<n_block, block_dim>>>(d_codes, d_vectors, d_permutes, this->P, this->W, this->K, this->units_per_band, bits_per_band ,n_vectors, LSTM_size);

    }
    
    
    
    
};



#endif /* LSH_WTA_h */
