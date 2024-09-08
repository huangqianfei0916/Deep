/*
 * Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
 * @Author: huangqianfei@tju.edu.cn
 * @Date: 2024-09-01 19:57:49
 * @Description: 
 */
#include <iostream>  
#include <string>  
#include <vector>  
  
// 函数用于获取UTF-8字符的长度（以字节为单位）  
// 注意：这里假设输入总是有效的UTF-8编码  
std::size_t utf8Length(const char* utf8Char) {  
    if ((unsigned char)utf8Char[0] <= 0x7F) return 1; // ASCII字符  
    else if ((utf8Char[0] & 0xE0) == 0xC0) return 2;   // 两个字节的UTF-8字符  
    else if ((utf8Char[0] & 0xF0) == 0xE0) return 3;   // 三个字节的UTF-8字符  
    else if ((utf8Char[0] & 0xF8) == 0xF0) return 4;   // 四个字节的UTF-8字符（通常用于特殊字符，不常见于中文字符）  
    else return 0; // 无效字符  
}  
  
// 遍历UTF-8编码的字符串  
void traverseUtf8String(const std::string& utf8Str) {  
    for (std::size_t i = 0; i < utf8Str.size();) {  
        std::size_t len = utf8Length(utf8Str.c_str() + i);  
        if (len > 0) {  
            std::cout << "字符: " << std::string(utf8Str.c_str() + i, len) << std::endl;  
            i += len;  
        } else {  
            // 处理无效字符，这里只是简单地跳过  
            std::cerr << "检测到无效字符，跳过..." << std::endl;  
            ++i;  
        }  
    }  
}  
  
int main() {  
    std::string utf8Str = "你好，世界world！";  
    traverseUtf8String(utf8Str);  
    return 0;  
}