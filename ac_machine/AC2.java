package com.hqf.ac;


import java.io.*;
import java.util.*;

import org.apache.commons.cli.*;


class ACNode{
    //    每个节点的字符
    char character;
    //    每个节点的孩子节点
    HashMap<Character, ACNode> children;
    //    每个节点的失配指针
    ACNode fail;
    //    每个节点是不是模式串终点节点，若是则为整个字符串，若不是则为None
    String patternStr;
    //    初始化节点
    public ACNode(char ch){
        character = ch;
        children = new HashMap<>();
        patternStr = "None";
        fail = null;
    }
}

public class AC {
    private ACNode root;
    private List<String> target_pattern;

    public AC(ArrayList<String> pattern){
        root = new ACNode(' ');
        target_pattern = pattern;
        buildTrieTree(pattern);
        System.out.println("trie tree build finish......");
//        checkTrieTree();
        buildFail();
        System.out.println("fail point build finish......");
    }

    public void buildFail(){
//        因为构建fail指针以来节点的父节点和父节点的fail节点，这里采用bfs构建
        LinkedList<ACNode> queue = new LinkedList<>();
//        先将root的孩子节点加入队列，同时root的孩子节点的fail全部指向root节点
        for (Map.Entry<Character, ACNode> entry : root.children.entrySet()) {
            entry.getValue().fail = root;
            queue.add(entry.getValue());
        }

        while(!queue.isEmpty()){
//            temp为父节点，tempNode是孩子节点
            ACNode temp = queue.poll();
            for (Map.Entry<Character, ACNode> entry : temp.children.entrySet()) {
                ACNode tempNode = entry.getValue();
                Character ch = tempNode.character;

                if(tempNode != null){
                    queue.add(tempNode);
                    ACNode failNode = temp.fail;
                    while (true){
                        if(failNode == null){
                            tempNode.fail = root;
                            break;
                        }
                        HashMap<Character, ACNode> fileChildMap = failNode.children;
                        if(fileChildMap.containsKey(ch)){
                            tempNode.fail = fileChildMap.get(ch);
                            break;
                        }else {
                            failNode = failNode.fail;
                        }
                    }
                }
            }
        }
    }

    public void search(String query, BufferedWriter bufferedWriter) throws IOException {
        ACNode current = root;
        int i = 0;
        while(i < query.length()){
            char ch = query.charAt(i);
            if(current.children.get(ch) != null){
                current = current.children.get(ch);

                if(current.patternStr != "None"){
                    bufferedWriter.write(current.patternStr + ":" + (i - current.patternStr.length()+1) + "\t");
                }

                ACNode temp = current;
                while(temp.fail != null && temp.fail.patternStr != "None"){
                    bufferedWriter.write(temp.fail.patternStr + ":" + (i - temp.fail.patternStr.length()+1) + "\t");
                    temp = temp.fail;
                }
                ++i;
            }else{
                current = current.fail;
                if(current == null){
                    current = root;
                    ++i;
                }
            }
        }
        bufferedWriter.newLine();
    }

    public void buildTrieTree(ArrayList<String> pattern){
        for(int i = 0; i < pattern.size(); ++i){
            ACNode current = root;
            String temp_pattern = pattern.get(i);

            for(int j = 0; j < temp_pattern.length(); ++j){
                char ch = temp_pattern.charAt(j);
                if(current.children.containsKey(ch)){
                    ACNode temp = current.children.get(ch);
                    current = temp;
                }else{
                    ACNode temp_node = new ACNode(ch);
                    current.children.put(ch, temp_node);
                    current = temp_node;
                }
            }
            current.patternStr = temp_pattern;
        }

    }

    public void checkTrieTree(){
        ACNode p = root;
        LinkedList<ACNode> queue = new LinkedList();
        queue.add(p);
        while(!queue.isEmpty()){
            ACNode temp = queue.poll();
            System.out.println(temp.character);
            for (Map.Entry<Character, ACNode> entry : temp.children.entrySet()) {
                if(entry != null){
                    System.out.println(entry.getKey() + ":" + entry.getValue().patternStr);
                    queue.add(entry.getValue());
                }
            }
        }
    }

    public static ArrayList<String> readFile(String path) throws IOException {
        ArrayList<String> list = new ArrayList<>();
        FileInputStream fis = new FileInputStream(path);
        InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
        BufferedReader br = new BufferedReader(isr);
        String line = "";
        while ((line = br.readLine()) != null) {
            list.add(line);
        }
        br.close();
        isr.close();
        fis.close();
        return list;
    }

    public static void main(String[] args) throws Exception {
        CommandLineParser parser = new DefaultParser();
        Options options = new Options();
        options.addOption("p", "parser", true, "parser set");
        options.addOption("q", "query", true, "query set");
        options.addOption("s", "save", true, "save file");
        CommandLine commandLine = parser.parse(options, args);

        String path = "";
        String textPath = "";
        String savePath = "";
        BufferedWriter bufferedWriter = null;

        if (commandLine.hasOption('p')) {
            path = commandLine.getOptionValue('p');
        }
        if (commandLine.hasOption('q')) {
            textPath = commandLine.getOptionValue('q');
        }
        if (commandLine.hasOption('s')) {
            savePath = commandLine.getOptionValue('s');
        }

        bufferedWriter = new BufferedWriter(new FileWriter(savePath));
        ArrayList<String> pattern = readFile(path);

        long start = System.currentTimeMillis();
        AC ac = new AC(pattern);

        FileInputStream fis = new FileInputStream(textPath);
        InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
        BufferedReader br = new BufferedReader(isr);
        String line = "";
        while ((line = br.readLine()) != null) {
            bufferedWriter.write(line + "\t");
            ac.search(line, bufferedWriter);
        }
        br.close();
        isr.close();
        fis.close();

        System.out.println("cost time: " + (System.currentTimeMillis() - start) + " ms");
        bufferedWriter.flush();
        bufferedWriter.close();

    }
}

