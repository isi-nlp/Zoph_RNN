//logger for writing to both 
#ifndef OUTPUT_LOGGER_H
#define OUTPUT_LOGGER_H

class OutputLogger {
public:
	bool log_output;
	std::string file_name;
	std::ofstream out_stream;
	OutputLogger () {
		log_output=false;
	}
	void SetOutputLogger (std::string file_name, bool log_output) {
		this->log_output = log_output;
		this->file_name = file_name;
		if(log_output) {
			out_stream.open(file_name.c_str());
		}
	}
};

template<typename T>
OutputLogger& operator<< (OutputLogger &out, T &t) {
	std::cout << t;
	if(out.log_output) {
		out.out_stream << t;
		out.out_stream.flush();
	}
	return out;
}

OutputLogger& operator<< (OutputLogger &out, int t) {
	std::cout << t;
	if(out.log_output) {
		out.out_stream << t;
		out.out_stream.flush();
	}
	return out;
}

OutputLogger& operator<< (OutputLogger &out, double t) {
	std::cout << t;
	if(out.log_output) {
		out.out_stream << t;
		out.out_stream.flush();
	}
	return out;
}

OutputLogger& operator<< (OutputLogger &out, float t) {
	std::cout << t;
	if(out.log_output) {
		out.out_stream << t;
		out.out_stream.flush();
	}
	return out;
}

OutputLogger& operator<< (OutputLogger &out,std::_Setprecision t) {
	std::cout << t;
	if(out.log_output) {
		out.out_stream << t;
		out.out_stream.flush();
	}
	return out;
}

#endif