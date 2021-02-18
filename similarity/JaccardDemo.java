package com.baidu.personal.similarity;

import java.util.*;

public class JaccardDemo {
    public static void main(String[] args) {
        calculateJaccard("dadadsf","dagdsxz");
    }

    public static float calculateJaccard(String str1, String str2) {
        Set<Character> s1 = new HashSet();
        Set<Character> s2 = new HashSet<>();
        for (int i = 0; i < str1.length(); i++) {
            s1.add(str1.charAt(i));
        }
        for (int i = 0; i < str2.length(); i++) {
            s2.add(str2.charAt(i));
        }
        float equalNum = 0;
        float mergeNum = 0;
        for (Character item1 : s1) {
            for (Character item2 : s2) {
                if (item1 == item2) {
                    equalNum++;
                }
            }
        }
        mergeNum = s1.size() + s2.size() - equalNum;
        float jaccard = equalNum / mergeNum;
        System.out.println(jaccard);
        return jaccard;
    }
}
