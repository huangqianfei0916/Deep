import sys
import re
import os

import queue

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


class ACNode(object):
    def __init__(self, ch):
        self._character = ch
        self._pattern_string = "None"
        self._children = {}
        self._fail = None


class AC(object):
    """AC自动机构造"""

    def __init__(self, pattern_list):
        self._root = ACNode(' ')

        self.build_tire_tree(pattern_list)
        # self.check_tree()
        print("trie tree build finish......")
        self.build_fail_point()
        print("fail point build finish......")

    def build_tire_tree(self, pattern_list):
        """
        build tire tree with ACNode
        """
        for pattern in pattern_list:
            current = self._root

            for ch in pattern:
                if ch in current._children:
                    current = current._children.get(ch)
                else:

                    temp_node = ACNode(ch)
                    current._children[ch] = temp_node
                    current = temp_node

            current._pattern_string = pattern

    def build_fail_point(self,):
        """
        bfs build fail point
        """
        q = queue.Queue()
        # 将根结点的孩子节点的fail指针指向根结点
        for key, value in self._root._children.items():
            value._fail = self._root
            q.put(value)

        while not q.empty():
            temp = q.get()

            for key, value in temp._children.items():
                ch = value._character

                q.put(value)
                fail_node = temp._fail

                while True:
                    if not fail_node:
                        value._fail = self._root
                        break

                    fail_child_map = fail_node._children

                    if ch in fail_child_map:
                        value._fail = fail_child_map[ch]
                        break
                    else:
                        fail_node = fail_node._fail
        
    def search(self, query):
        """
        search
        """
        res = []
        current = self._root
        i = 0
        while i < len(query):
            ch = query[i]
            if ch in current._children and current._children[ch]:
                current = current._children[ch]
                if current._pattern_string != "None":
                    res.append(current._pattern_string)

                temp = current
                while temp._fail and temp._fail._pattern_string != "None":
                    res.append(temp._fail._pattern_string)
                    temp = temp._fail
                i += 1
            else:
                current = current._fail
                if not current:
                    current = self._root
                    i += 1
        return res

    def check_tree(self,):
        p = self._root
        q = queue.Queue()
        q.put(p)
        while not q.empty():
            temp = q.get()
            for key, value in temp._children.items():
                print(key + ":" + value._pattern_string)
                q.put(value)

    

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("参数错误")
        exit(1)

    pattern_file = sys.argv[1]
    

    f_r = open(pattern_file)
    pattern_list = []
    for pattern in f_r:
        pattern = pattern.strip()
        pattern_list.append(pattern)

    ac = AC(pattern_list)

    res = ac.search("打伞的鱼coco加盟彩之丽女装")
    print(res)