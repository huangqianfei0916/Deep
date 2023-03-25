/*
 * @Author: huangqianfei
 * @Date: 2023-03-25 12:37:18
 * @LastEditTime: 2023-03-25 21:26:50
 * @Description: word2vec 的cpp实现
 * 参考博客：https://blog.csdn.net/So_that/article/details/103146219?spm=1001.2014.3001.5502
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>

#include <cmath>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary
typedef float real;

struct vocab_word
{
    long long cn;
    int *point;
    char *word, *code, codelen;
};


std::string OUTPUT_FILE = "./output_file.txt";

struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
long long vocab_max_size = 1000, emb_size = 100, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;

std::vector<real> word_emb;
std::vector<real> syn1;
std::vector<real> neg_emb;

clock_t start;
int hs = 0, negative = 5;
const int table_size = 1e8;

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


// 预计算sigmoid
void pre_calcate_sigmoid(std::vector<real>& exp_table) {

    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        // Precompute the exp() table
        exp_table[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); 
        // Precompute f(x) = x / (x + 1)
        exp_table[i] = exp_table[i] / (exp_table[i] + 1);                   
    }

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



void read_file(std::string& train_file) {
    std::ifstream f_r;
    f_r.open(train_file);
    std::string temp_line;
    while (getline(f_r, temp_line)) {
        std::vector<std::string> tokens;
        split(temp_line, tokens);
        for (auto& token : tokens) {
            ++vocab_map[token];
        }
    }

}


void init_net() {
    word_emb.resize(vocab_map.size() * emb_size);
    int vocab_size = vocab_map.size();

    if (hs) {
        // 才有hs优化算法时，初始化syn1
        syn1.resize(vocab_size * emb_size);
    }

    if (negative > 0) {
        // 负采样初始化
        neg_emb.resize(vocab_size * emb_size)
    }

    //初始化word_emb数组（也就是词向量） 并不是0，范围是[-0.5/m,0.5/m],其中m词向量的维度。
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < emb_size; j++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            word_emb[i * emb_size + j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / emb_size;
        }
    }

    // 构建huffman 树
    
}


// 训练
void train(std::string& train_file) {

    read_file(train_file);
    init_net();


}



int main(int argc, char *argv[]) {

    // if (argc == 1) {
    //     std::cout << "word2vec cpp useage" << std::endl;
    //     std::cout << "wait" << std::endl;
    //     std::cout << "-------------------" << std::endl;
    //     return 0;
    // }

    int index = 0;
    index = args_parse("-train", argc, argv);
    std::string train_file = argv[index + 1];

    std::vector<real> exp_table(EXP_TABLE_SIZE);
    pre_calcate_sigmoid(exp_table);

    train(train_file);


    return 1;
}
