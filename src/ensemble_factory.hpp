
template<typename dType>
ensemble_factory<dType>::ensemble_factory(std::vector<std::string> weight_file_names,int num_hypotheses,int beam_size, dType min_decoding_ratio,\
                                          dType penalty, int longest_sent,bool print_score,std::string decoder_output_file,
                                          std::vector<int> gpu_nums,dType max_decoding_ratio, int target_vocab_size,global_params &params)
{

  //get the target vocab from the first file
  this->target_vocab_size = target_vocab_size;
  this->max_decoding_ratio = max_decoding_ratio;
  this->longest_sent = longest_sent;
  this->p_params = &params;

  this->interactive = params.interactive;
  this->interactive_line = params.interactive_line;


  //to make sure beam search does halt
  if(beam_size > (int)std::sqrt(target_vocab_size) ) {
    beam_size = (int)std::sqrt(target_vocab_size);
  }

  //fileh = new file_helper_decoder(input_file_name,num_lines_in_file,longest_sent);
  std::ifstream temp_input;
  temp_input.open(params.decode_temp_files[0]);
  get_file_stats_source(num_lines_in_file,temp_input);
  temp_input.close();

  model_decoder = new decoder<dType>(beam_size,target_vocab_size,start_symbol,end_symbol,longest_sent,min_decoding_ratio,
                                     penalty,decoder_output_file,num_hypotheses,print_score, params);
	
  model_decoder->print_beam = params.print_beam;
    
  //initialize all of the models
  models.reserve(weight_file_names.size());
  for(int i=0; i < weight_file_names.size(); i++) {
    // avoid the deleting-temporaries behavior of push_back
    // emplace_back passes arguments on to the object constructor. For reference, the original
    // push back is left below in comments.
    models.emplace_back(gpu_nums[i],beam_size,params.model_names[i],params.model_names_multi_src[i],
                        params.decode_temp_files[i],params.decode_temp_files_additional[i],longest_sent,params);
    //models.push_back( decoder_model_wrapper<dType>(gpu_nums[i],beam_size,params.model_names[i],params.model_names_multi_src[i],
    //                                               params.decode_temp_files[i],params.decode_temp_files_additional[i],longest_sent,params));

  }


  //check to be sure all models have the same target vocab size and vocab indicies and get the target vocab size
  this->target_vocab_size = models[0].target_vocab_size;
  // TODO: i not used; something seems wrong here...
  for(int i=0; i< models.size(); i++) {
    if(models[0].target_vocab_size != target_vocab_size) {
      BZ_CUDA::logger << "ERROR: The target vocabulary sizes are not all the same for the models you wanted in your ensemble\n";
      exit (EXIT_FAILURE);
    }
  }

  //resise the outputdist that gets sent to the decoder
  outputdist.resize(target_vocab_size,beam_size);
  normalization.resize(1,beam_size);

}

template<typename dType>
void ensemble_factory<dType>::decode_file() {
  if (this->interactive){
    if (this->interactive_line){
      decode_file_interactive_line();
    } else {
      decode_file_interactive();
    }
  } else {
    decode_file_batch();
  }
}

template<typename dType>
void ensemble_factory<dType>::decode_file_interactive_line() {
  bool right_after_encoding = true;
  int k = 1;
  model_decoder->num_hypotheses = k;

  // for words and fsaline, they can be in the middle of the decoding,
  // suppose words = [w1, w2, w3] or fsaline will decode as [w1, w2, w3],
  // both of the two funcs needs to prepare the follwing two things:
  // 1. init the pre_target_states.c_t_pre/h_t_pre as h_2 ( h2 = lstm(w2,h1) )
  // 2. init the h_current_indicies = [w3] * beam_size;
    

  while (true) {
    // 1. source <source_file>  -> [END]
    // 2. words <words> -> [END]
    // 3. fsa <fsa_file> encourage_list_files:enc1.txt,enc2.txt encourage_weights:1.0,-1.0 repetition:0.0 alliteration:0.0 wordlen:0.0 -> [END] : as normal
    // 4. fsaline <fsa_file> encourage_list_files:enc1.txt,enc2.txt encourage_weights:1.0,-1.0 repetition:0.0 alliteration:0.0 wordlen:0.0 -> [END]: as noraml, but at the end, move corresponding ct and ht to all beams.
        
        
    std::cout<<"Please input <source/words/fsa/fsaline> <source_file/words/fsa_file>\n";
    std::cout.flush();
    // read input
    // input format:
    // <k> <source_file> <fsa_file>
        
    std::string source_file = "";
    std::string fsa_file = "";
    std::string line;
    //std::cout<<"line=" << line <<"\n";
        
        
    std::getline(std::cin, line);
    std::vector<std::string> ll = split(line,' ');
        
    std::string action = ll[0];
        
    //std::cout<<"line=" << line <<"\n";
        
    if (action == "source") {
      source_file = ll[1];
            
      input_file_prep input_helper;
      input_helper.integerize_file_kbest(p_params->model_names[0],source_file,p_params->decode_temp_files[0],
                                         p_params->longest_sent,p_params->target_vocab_size,false,"NULL", p_params->legacy_model);
            
      int num_lines_in_file = 1;
            
      if(models[0].fileh != NULL){
        delete models[0].fileh;
        models[0].fileh = NULL;
      }
      models[0].fileh = new file_helper_decoder(p_params->decode_temp_files[0],num_lines_in_file,p_params->longest_sent,p_params->char_params,p_params->char_params.char_test_file);
            
      this->num_lines_in_file = num_lines_in_file;
            
            
      //copy the indicies all the models on the gpu
      //in memcpy_vocab_indicies
      for(int j=0; j < models.size(); j++) {
        models[j].memcpy_vocab_indicies();
      }
            
      devSynchAll();
            
      //init decoder
      model_decoder->init_decoder();
      //run forward prop on the source
      for(int j=0; j < models.size(); j++) {
        models[j].forward_prop_source();
      }

      std::cout<<"[END]\n";
      std::cout.flush();
            
      right_after_encoding = true;
            
    } else if (action == "words") {
      std::vector<int> word_indices;
            
      int curr_index = 0;
            
      if (right_after_encoding){
        curr_index = 0;
      } else {
        curr_index = 1;
      }

            
      for (int i = 1; i < ll.size(); i +=1 ){
        std::string word = ll[i];
        int word_index = 2; // <UNK>
        if (model_decoder->tgt_mapping.count(word) > 0){
          word_index = model_decoder->tgt_mapping[word];
        }
        word_indices.push_back(word_index);
      }
            
            
            
      for (int i = 0; i< word_indices.size() ; i ++){
                

        std::cout<< "WI: "<< model_decoder->h_current_indices[0] << "\n";
                
        for(int j=0; j < models.size(); j++) {
          models[j].forward_prop_target(curr_index+i,model_decoder->h_current_indices);
          models[j].target_copy_prev_states();
        }
                    
        int word_index = word_indices[i];
                    
        for (int j=0 ; j< model_decoder->beam_size; j++){
          model_decoder->h_current_indices[j] = word_index;
        }

                
      }
            
            
      std::cout<<"[END]\n";
      std::cout.flush();
            
      right_after_encoding = false;
            
    } else if (action == "fsa") {
      fsa_file = ll[1];
            
      model_decoder->init_fsa_interactive(fsa_file);
            
      model_decoder->init_decoder(models[0].fileh->sentence_length, right_after_encoding);
            
      //process the encourage list file and encourage weight;
      std::vector<std::string> encourage_list_files;
      std::vector<float> encourage_weights;
      float repetition_weight = 0.0;
      float alliteration_weight = 0.0;
      float wordlen_weight = 0.0;

      for (int i = 2; i < ll.size(); i +=1){
        std::vector<std::string> pair = split(ll[0],':');
        if (pair[0] == "repetition"){
          repetition_weight = std::stof(pair[1]);
        }
        if (pair[0] == "alliteration"){
          alliteration_weight = std::stof(pair[1]);
        }
        if (pair[0] == "wordlen"){
          wordlen_weight = std::stof(pair[1]);
        }
        if (pair[0] == "encourage_list_files"){
          encourage_list_files = split(pair[1],',');
        }
        if (pair[0] == "encourage_weights"){
          std::vector<std::string> weight_strs = split(pair[1],',');
          for (const std::string weight_str: weight_strs)
	    {
	      encourage_weights.push_back(std::stof(weight_str));
	    }
        }
      }
            
      //process the encourage list file and encourage weight;
      model_decoder->init_encourage_lists(encourage_list_files,encourage_weights);
      //process repetition (>0 more repetition; <0 less repetition)
      model_decoder->interactive_repeat_penalty = repetition_weight;
      //process alliteration
      model_decoder->alliteration_weight = alliteration_weight;
      //process wordlen weight
      model_decoder->wordlen_weight = wordlen_weight;

            
            
      decode_file_line(right_after_encoding,false);
            
      //read output and print into stdout;
      input_file_prep input_helper;
      input_helper.unint_file(p_params->model_names[0],p_params->decoder_output_file,p_params->decoder_final_file,false,true);
            
      std::string file_line;
      std::ifstream infile(p_params->decoder_final_file);
            
      while (std::getline(infile,file_line)){
        std::cout << file_line << "\n";
      }
            
      std::cout<< "[END]\n";
      std::cout.flush();
            
      infile.close();
            
      // close the kbest.txt
      model_decoder->output.close();
      model_decoder->output.open(model_decoder->output_file_name.c_str());
            
      right_after_encoding = false;
            
            
    } else if (action == "fsaline") {
            
      fsa_file = ll[1];
            
      model_decoder->init_fsa_interactive(fsa_file);
            
      model_decoder->init_decoder(models[0].fileh->sentence_length, right_after_encoding);
            
      //process the encourage list file and encourage weight;
      std::vector<std::string> encourage_list_files;
      std::vector<float> encourage_weights;
      float repetition_weight = 0.0;
      float alliteration_weight = 0.0;
      float wordlen_weight = 0.0;
            
      for (int i = 2; i < ll.size(); i +=1){
        std::vector<std::string> pair = split(ll[i],':');
        if (pair[0] == "repetition"){
          repetition_weight = std::stof(pair[1]);
        }
        if (pair[0] == "alliteration"){
          alliteration_weight = std::stof(pair[1]);
        }
        if (pair[0] == "wordlen"){
          wordlen_weight = std::stof(pair[1]);
        }
        if (pair[0] == "encourage_list_files"){
          encourage_list_files = split(pair[1],',');
        }
        if (pair[0] == "encourage_weights"){
          std::vector<std::string> weight_strs = split(pair[1],',');
          for (const std::string weight_str: weight_strs)
	    {
	      encourage_weights.push_back(std::stof(weight_str));
	    }
        }
      }
            
      //process the encourage list file and encourage weight;
      model_decoder->init_encourage_lists(encourage_list_files,encourage_weights);
      //process repetition (>0 more repetition; <0 less repetition)
      model_decoder->interactive_repeat_penalty = repetition_weight;
      //process alliteration
      model_decoder->alliteration_weight = alliteration_weight;
      //process wordlen weight
      model_decoder->wordlen_weight = wordlen_weight;
            
      model_decoder->model = models[0].model;
      decode_file_line(right_after_encoding,true);
            
      //read output and print into stdout;
      input_file_prep input_helper;
      input_helper.unint_file(p_params->model_names[0],p_params->decoder_output_file,p_params->decoder_final_file,false,true);
            
      std::string file_line;
      std::ifstream infile(p_params->decoder_final_file);
            
      while (std::getline(infile,file_line)){
        std::cout << file_line << "\n";
      }
            
      std::cout<< "[END]\n";
      std::cout.flush();
            
      infile.close();
            
      // close the kbest.txt
      model_decoder->output.close();
      model_decoder->output.open(model_decoder->output_file_name.c_str());
            
      right_after_encoding = false;
            
    }
  }
    
}

template<typename dType>
void ensemble_factory<dType>::decode_file_line(bool right_after_encoding, bool end_transfer) {
  // right_after_encoding = true, means the system is never decoding a word,
  //
  bool pre_end_transfer = model_decoder->end_transfer;
  model_decoder->end_transfer = end_transfer;
    
    
  int start_index = 1;
  if (right_after_encoding){
    start_index = 0;
  }
    
    
  //run the forward prop of target
  for(int curr_index=0; curr_index < std::min( (int)(max_decoding_ratio*models[0].fileh->sentence_length) , longest_sent-2 ); curr_index++) {
        
    //std::cout << "WI:" << model_decoder->h_current_indices[0]<<"\n";

    for(int j=0; j < models.size(); j++) {
      // curr_index: whether it's 0 or non-0. Doesn't matter if it's 1 or 2 or 3.
      // &c_t_pre = &pre_state ; c_t = f(c_t_pre)
      models[j].forward_prop_target(curr_index+start_index,model_decoder->h_current_indices);
    }
        
        
    //now ensemble the models together
    ensembles_models();
        
    //run decoder for this iteration
    model_decoder->expand_hypothesis(*p_outputdist,curr_index,BZ_CUDA::viterbi_alignments,models[0].h_outputdist);
    //swap the decoding states
    for(int j=0; j<models.size(); j++) {
      // here the curr_index doesn't matter
      // c_t_swap = swa(c_t)
      models[j].swap_decoding_states(model_decoder->new_indicies_changes,curr_index);
      // pre_stat = c_t_swap
      models[j].target_copy_prev_states();
    }
        
    //for the scores of the last hypothesis
        
    //std::cout<<model_decoder->invalid_number << " "<< model_decoder->beam_size<<"\n";
    if (model_decoder->invalid_number == model_decoder->beam_size){
      break;
    }
        
  }
    
  //now run one last iteration
  // in case next step, we can generate <EOF>
  std::cout << "WI:" << model_decoder->h_current_indices[0]<<"\n";
    
  for(int j=0; j < models.size(); j++) {
    models[j].forward_prop_target(1,model_decoder->h_current_indices);
  }
  //output the final results of the decoder
  ensembles_models();
  model_decoder->finish_current_hypotheses(*p_outputdist,BZ_CUDA::viterbi_alignments);
  model_decoder->output_k_best_hypotheses(models[0].fileh->sentence_length);
  //model_decoder->print_current_hypotheses();
  model_decoder->end_transfer = pre_end_transfer;
}




template<typename dType>
void ensemble_factory<dType>::decode_file_interactive() {
  // language model is not ready for the new input format;
    
  while (true) {
    std::cout<<"Please input k:<k> source_file:<source_file> fsa_file:<fsa_file> repetition:<repetition_weight> alliteration:<alliteration_weight> wordlen:<wordlen_weight> encourage_list_files:<file1>,<file2> encourage_weights:<weight1>,<weight2>\n";
        
    std::cout.flush();
    // read input
    // input format:
    // <k> <source_file> <fsa_file>
        
    std::string line;
    std::getline(std::cin, line);
    std::vector<std::string> ll = split(line,' ');
        
    int k = 1;
    std::string source_file = "";
    std::string fsa_file = "";
    std::vector<std::string> encourage_list_files;
    std::vector<float> encourage_weights;
    float repetition_weight = 0.0;
    float alliteration_weight = 0.0;
    float wordlen_weight = 0.0;
        
    for (const std::string opt : ll){
      std::vector<std::string> pair = split(opt,':');
      if (pair[0] == "k"){
        k = std::stoi(pair[1]);
      }
      if (pair[0] == "source_file")
	{
	  source_file = pair[1];
	}
      if (pair[0] == "fsa_file")
	{
	  fsa_file = pair[1];
	}
      if (pair[0] == "repetition"){
        repetition_weight = std::stof(pair[1]);
      }
      if (pair[0] == "alliteration"){
        alliteration_weight = std::stof(pair[1]);
      }
      if (pair[0] == "wordlen"){
        wordlen_weight = std::stof(pair[1]);
      }
      if (pair[0] == "encourage_list_files"){
        encourage_list_files = split(pair[1],',');
      }
      if (pair[0] == "encourage_weights"){
        std::vector<std::string> weight_strs = split(pair[1],',');
        for (const std::string weight_str: weight_strs)
	  {
	    encourage_weights.push_back(std::stof(weight_str));
	  }
      }
            
    }
        
    //process the number of hypothesis
    model_decoder->num_hypotheses = k;
        
    //process the input file
    //
    input_file_prep input_helper;
    input_helper.integerize_file_kbest(p_params->model_names[0],source_file,p_params->decode_temp_files[0],
                                       p_params->longest_sent,p_params->target_vocab_size,false,"NULL", p_params->legacy_model);
        
    int num_lines_in_file = 1;
        
    if(models[0].fileh != NULL){
      delete models[0].fileh;
      models[0].fileh = NULL;
    }
    models[0].fileh = new file_helper_decoder(p_params->decode_temp_files[0],num_lines_in_file,p_params->longest_sent,p_params->char_params,p_params->char_params.char_test_file);
        
    this->num_lines_in_file = num_lines_in_file;
        
        
    //process the input fsa file
    if (fsa_file != ""){
      //fsa
      model_decoder->init_fsa_interactive(fsa_file);
    }
        
    //process the encourage list file and encourage weight;
    model_decoder->init_encourage_lists(encourage_list_files,encourage_weights);
        
    //process repetition (>0 more repetition; <0 less repetition)
    model_decoder->interactive_repeat_penalty = repetition_weight;
    //process alliteration
    model_decoder->alliteration_weight = alliteration_weight;
    //process wordlen weight
    model_decoder->wordlen_weight = wordlen_weight;
        
        
    decode_file_batch();
        
    //read output and print into stdout;
    input_helper.unint_file(p_params->model_names[0],p_params->decoder_output_file,p_params->decoder_final_file,false,true);
        
    std::string file_line;
    std::ifstream infile(p_params->decoder_final_file);
        
    while (std::getline(infile,file_line)){
      std::cout << file_line << "\n";
    }
        
    std::cout<< "[END]\n";
    std::cout.flush();
        
    infile.close();
        
    // close the kbest.txt
    model_decoder->output.close();
    model_decoder->output.open(model_decoder->output_file_name.c_str());
        
  }
}


template<typename dType>
void ensemble_factory<dType>::decode_file_batch() {
	
  for(int i = 0; i < num_lines_in_file; i++) {
        
    models[0].model->timer.start("total");
        
    models[0].model->timer.start("memcpy_vocab_indicies");

    BZ_CUDA::logger << "Decoding sentence: " << i << " out of " << num_lines_in_file << "\n";
    //fileh->read_sentence(); //read sentence from file

    //copy the indicies all the models on the gpu
    //in memcpy_vocab_indicies
    for(int j=0; j < models.size(); j++) {
      models[j].memcpy_vocab_indicies();
    }
    devSynchAll();

    models[0].model->timer.end("memcpy_vocab_indicies");

    //init decoder
    model_decoder->init_decoder();
        
    models[0].model->timer.start("forward_source");
    //run forward prop on the source
    for(int j=0; j < models.size(); j++) {
      models[j].forward_prop_source();
    }
        
    models[0].model->timer.end("forward_source");
        
    int last_index = 0;

    //for dumping hidden states we can just return
    if(BZ_STATS::tsne_dump) {
      continue;
    }

    //run the forward prop of target
    //BZ_CUDA::logger << "Source length bi: " << models[0].source_length_bi << "\n";
    int source_length = std::max(models[0].source_length,models[0].source_length_bi);
        
        
    // prepare the target set vocabulary;
        
    models[0].model->timer.start("shrink_target_vocab");

    for(int j=0; j < models.size(); j++) {
      models[j].prepare_target_vocab_set();
      models[j].before_target_vocab_shrink();
    }

    models[0].model->timer.end("shrink_target_vocab");
        
    for(int curr_index=0; curr_index < std::min( (int)(max_decoding_ratio*source_length) , longest_sent-2 ); curr_index++) {

            
      models[0].model->timer.start("forward_target");
      if (p_params->target_vocab_policy == 2){
        // mapping current indicies
        // JM: single model assumption?
        for (int i = 0; i<models[0].beam_size; i++){
          int mapped_index = model_decoder->h_current_indices[i];
          model_decoder->h_current_indices_original[i] = models[0].h_new_vocab_index[mapped_index];
        }

        for(int j=0; j < models.size(); j++) {
          // do current d
          models[j].forward_prop_target(curr_index,model_decoder->h_current_indices_original);
          //now take the viterbi alignments
        }
                
      } else {
                
        for(int j=0; j < models.size(); j++) {
          // do current d
          models[j].forward_prop_target(curr_index,model_decoder->h_current_indices);
          //now take the viterbi alignments
        }
      }
      models[0].model->timer.end("forward_target");

      //now ensemble the models together
      //this also does voting for unk-replacement
      //	BZ_CUDA::logger << "Source length: " << source_length << "\n";

      models[0].model->timer.start("ensembles_models");
      ensembles_models();
      models[0].model->timer.end("ensembles_models");

      models[0].model->timer.start("expand");
            
            
      //run decoder for this iteration
      if (p_params->target_vocab_policy == 3){
        model_decoder->nnz = models[0].nnz; // for LSH_WTA
        int *temp = models[0].model->softmax->get_h_rowIdx();
        //std::cout << "temp: "<< temp;
        model_decoder->h_rowIdx = temp;
      }
			
      // TODO: this change looks suspicious; model 0 assumption...
      model_decoder->expand_hypothesis(*p_outputdist,curr_index,BZ_CUDA::viterbi_alignments,models[0].h_outputdist);

      models[0].model->timer.end("expand");
            
      models[0].model->timer.start("swap_decoding_states");


      //swap the decoding states
      for(int j=0; j<models.size(); j++) {
        models[j].swap_decoding_states(model_decoder->new_indicies_changes,curr_index);
        models[j].target_copy_prev_states();
      }

      models[0].model->timer.end("swap_decoding_states");

      //for the scores of the last hypothesis
      last_index = curr_index;
            
      if (model_decoder->invalid_number == model_decoder->beam_size){
        break;
      }

            
    }

    models[0].model->timer.start("forward_target");
    //now run one last iteration
    if (p_params->target_vocab_policy == 2){
      // mapping current indicies
      for (int i = 0; i<models[0].beam_size; i++){
        int mapped_index = model_decoder->h_current_indices[i];
        model_decoder->h_current_indices_original[i] = models[0].h_new_vocab_index[mapped_index];
      }
      for(int j=0; j < models.size(); j++) {
        // do current d
        models[j].forward_prop_target(last_index+1,model_decoder->h_current_indices_original);
        //now take the viterbi alignments
      }
            
    } else {

      for(int j=0; j < models.size(); j++) {
        models[j].forward_prop_target(last_index+1,model_decoder->h_current_indices);
      }
    }
        
    models[0].model->timer.end("forward_target");

        
    //output the final results of the decoder
    models[0].model->timer.start("forward_target");
    ensembles_models();
    models[0].model->timer.end("forward_target");


    models[0].model->timer.start("output_k_best");

    model_decoder->finish_current_hypotheses(*p_outputdist,BZ_CUDA::viterbi_alignments);
    model_decoder->output_k_best_hypotheses(source_length, models[0].h_new_vocab_index, (p_params->target_vocab_policy == 1 || p_params->target_vocab_policy == 2));
        
    models[0].model->timer.end("output_k_best");
		
    //model_decoder->print_current_hypotheses();
        
    // after target_vocab_shrink
    models[0].model->timer.start("shrink_target_vocab");

    for(int j=0; j < models.size(); j++) {
      models[j].after_target_vocab_shrink();
    }
    models[0].model->timer.end("shrink_target_vocab");

        
    models[0].model->timer.end("total");

  }
    
  models[0].model->timer.report();
  models[0].model->timer.clear();

}

template<typename dType>
void ensemble_factory<dType>::ensembles_models() {
  int num_models = models.size();
    
  if (num_models == 1){
    //outputdist = models[0].outputdist;
    p_outputdist = &models[0].outputdist;
  } else {
    for(int i=0; i<outputdist.rows(); i++) {
      for(int j=0; j< outputdist.cols(); j++) {
        double temp_sum = 0;
        for(int k=0; k<models.size(); k++) {
          temp_sum+=models[k].outputdist(i,j);
        }
        outputdist(i,j) = temp_sum/num_models;
      }
    }
        
    normalization.setZero();
        
    for(int i=0; i<outputdist.rows(); i++) {
      normalization+=outputdist.row(i);
    }
    for(int i=0; i<outputdist.rows(); i++) {
      outputdist.row(i) = (outputdist.row(i).array()/normalization.array()).matrix();
    }
        
    p_outputdist = & outputdist;
  }

  //now averaging alignment scores for unk replacement
  if(BZ_CUDA::unk_replacement) {
    //average the scores
    for(int i=0; i<models[0].longest_sent;i++) {
      for(int j=0; j<models[0].beam_size; j++) {
        dType temp_sum = 0;
        for(int k=0; k<models.size(); k++) {
          temp_sum+=models[k].viterbi_alignments_scores[IDX2C(i,j,models[0].longest_sent)];
        }
        BZ_CUDA::alignment_scores[IDX2C(i,j,models[0].longest_sent)] = temp_sum;
      }
    }

    // std::cout << "-------------------------------------------\n";
    // for(int i=0; i<models[0].longest_sent;i++) {
    // 	for(int j=0; j<models[0].beam_size; j++) {
    // 		std::cout << BZ_CUDA::alignment_scores[IDX2C(i,j,models[0].longest_sent)] << " ";
    // 	}
    // 	std::cout << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "-------------------------------------------\n\n";
    //choose the max and fill in BZ_CUDA::viterbi_alignments
    for(int i=0; i<models[0].beam_size; i++) {
      dType max_val = 0;
      int max_index = -1;
      for(int j=0; j<models[0].longest_sent; j++) {
        dType temp_val = BZ_CUDA::alignment_scores[IDX2C(j,i,models[0].longest_sent)];
        if(temp_val > max_val) {
          max_val = temp_val;
          max_index = j;
        }
      }
      // if(max_index==-1) {
      // 	std::cout << "ERROR: max_index is still -1, so all values are zero\n";
      // }
      BZ_CUDA::viterbi_alignments[i] = max_index;
    }
  }
}

