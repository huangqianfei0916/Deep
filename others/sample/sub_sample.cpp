// useage: ./sample -i query.txt -o output

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>

#include <cmath>

typedef float real;

unsigned long long next_random = 1;
long long  tw = 0;
real sample = 1e-2;

std::map<std::string, int> vocab_map;

// 参数解析
int args_parse(std::string flag, int argc, char *argv[]) {

    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) {
            if (i == argc - 1) {
                exit(1);
            }
            return i;
        }
    }
    return -1;
}


// 字符串切割
void split(const std::string s, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    int lastPos = s.find_first_not_of(delimiters);
    int pos = s.find_first_of(delimiters);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}


// 统计词频
long long read_file(std::string& train_file, int min_count = 1) {
    std::ifstream f_r;
    f_r.open(train_file);
    std::string temp_line;
    while (getline(f_r, temp_line)) {
        ++vocab_map[temp_line];
    }

    for (auto& [key, value] : vocab_map) {
        if (value < min_count) {
            continue;
        }

        tw += value;
    }
    return tw;
}


// 高频词的采样，词频越高，被舍弃的概率越大
int word_sample(float cn) {
    int flag = 1;
    real word_pro = (sqrt(cn / (sample * tw)) + 1) * (sample * tw) / cn;
    next_random = next_random * (unsigned long long)25214903917 + 11;

    // std::cout << word_pro << "---" << (next_random & 0xFFFF) / (real)65536 << std::endl;

    if (word_pro < (next_random & 0xFFFF) / (real)65536) {
        flag = -1;
    }
    return flag;
}


// 采样
void shuffle_sample(std::string& train_file, std::vector<std::string>& lines) {

    std::ifstream f_r;
    f_r.open(train_file);
    std::string temp_line;

    while (getline(f_r, temp_line)) {

        int cn = vocab_map[temp_line];
        int flag = word_sample(cn);

        if (flag > 0) {
            lines.push_back(temp_line);
        }
    }

}


int main(int argc, char* argv[]) {

    int index = 0;
    index = args_parse("-i", argc, argv);
    std::string train_file = argv[index + 1];

    index = args_parse("-o", argc, argv);
    std::string output_file = argv[index + 1];
    std::cout << output_file << std::endl;
    long long tw = read_file(train_file);

    std::vector<std::string> lines;
    shuffle_sample(train_file, lines);

    std::ofstream f_w;
    f_w.open(output_file);

    for (auto& line : lines) {
        f_w << line << "\n";
    }

    return 0;  
}