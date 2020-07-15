参考链接：https://www.cnblogs.com/jpcflyer/p/11180263.html

* 将数据构成一个有向网络，这个网络存在出度和入度。
* 一个网页的影响力 = 所有入链集合的页面的加权影响力之和，
* [pagerank1](pagerank1.png)
* [pagerank2](pagerank2.png)
*  等级泄露（Rank Leak）：如果一个网页没有出链，就像是一个黑洞一样，吸收了其他网页的影响力而不释放，最终会导致其他网页的 PR 值为 0。
* 等级沉没（Rank Sink）：如果一个网页只有出链，没有入链（如下图所示），计算的过程迭代下来，会导致这个网页的 PR 值为 0（也就是不存在公式中的 V）
* 等级泄露和等级沉没解决方案：
* [pagerank3](pagerank3.png)