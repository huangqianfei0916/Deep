/*
 * @Author: huangqianfei
 * @Date: 2023-03-26 16:46:23
 * @LastEditTime: 2023-04-16 18:48:46
 * @Description: 
 */

#include <iostream>
#include <vector>


// in->hidden->hs|sample



void run_cbow(std::vector<float>& word_emb, 
       int window_size, 
       std::vector<std::vector<int>>& sentences) {

    int cw = 0;
    int sentence_position = 0;
    std::vector<float> neu1(window_size);
    int layer1_size = 100, index = 0;
    unsigned long long next_random = next_random * (unsigned long long)25214903917 + 11;
    int b = next_random % window_size;

    for (int i = 0; i < sentences.size(); ++i) {
        std::vector<int> sen = sentences[i];
        int sentence_length = sen.size();

        for (int a = b; a < window_size * 2 + 1 - b; ++a) {
            if (a != window_size) {
                index = sentence_position - window_size + a;
                if (index < 0) continue;
                if (index >= sentence_length) continue;
                int last_word = sen[index];
                if (last_word == -1) continue;
                for (index = 0; index < layer1_size; index++) neu1[index] += word_emb[index + last_word * layer1_size];
                cw++;
            }

        }

    }



    
}