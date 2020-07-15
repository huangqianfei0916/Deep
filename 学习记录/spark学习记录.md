### RDD，DataFrame和DataSet之间的区别

### SparkContext 和 SparkConf以及SparkSession

* 任何Spark程序都是SparkContext开始的，SparkContext的初始化需要一个SparkConf对象；
* SparkConf包含了Spark集群配置的各种参数。
* 初始化后，就可以使用SparkContext对象所包含的各种方法来创建和操作RDD和共享变量。

```scala
val conf = new SparkConf().setMaster("master").setAppName("appName")
val sc = new SparkContext(conf)
或者
val sc = new SparkContext("master","appName")
```
* SparkSession： SparkSession实质上是SQLContext和HiveContext的组合（未来可能还会加上StreamingContext），
* 所以在SQLContext和HiveContext上可用的API在SparkSession上同样是可以使用的。
* SparkSession内部封装了sparkContext，所以计算实际上是由sparkContext完成的。

```scala
val conf = new SparkConf().setMaster("local[2]").setAppName("NetworkWordCount")
val ssc = new StreamingContext(conf, Seconds(1))
```


### 关于spark streaming的使用

* 1 定义上下文；
* 2 通过创建DStream定义输入源；
* 3 对DStream应用一些列流操作；
* 4 开始接收数据并使用进行处理streamingContext.start()；
* 5 等待处理结束或者遇到错误，streamingContext.awaitTermination()；
* 6 手动停止处理streamingContext.stop()；

### 注意点
* 一旦启动上下文，就无法设置新的流计算或将其添加到该流计算中；
* 上下文一旦停止，就无法重新启动。
* JVM中只能同时激活一个StreamingContext。
* StreamingContext上的stop（）也会停止SparkContext。要仅停止的StreamingContext，设置可选的参数stop()叫做stopSparkContext假。
* 只要在创建下一个StreamingContext之前停止了上一个StreamingContext（无需停止SparkContext），就可以将SparkContext重用于创建多个StreamingContext。



*** 
* 每个DStream对应一个接收器；
* 可以创建多个Dstream对应多个接收器；来同时处理多个数据流；但要注意本地时线程数要足够；

* ssc.socketTextStream(...)从socket接受；
* streamingContext.fileStream[KeyClass, ValueClass, InputFormatClass](dataDirectory)文件流；
* 使用kafka需要相应的jar；
* streamingContext.queueStream(queueOfRDDs)。推送到队列中的每个RDD将被视为DStream中的一批数据，并像流一样进行处理。
* 自定义接收器；


### map,flatmap
* map是对每一行进行处理；flatmap就是在map之后再进行flat
```scala
val rdd = sc.parallelize(List("coffee panda","happy panda","happiest panda party"))
rdd.map(x=>x).collect
res9: Array[String] = Array(coffee panda, happy panda, happiest panda party)
rdd.flatMap(x=>x.split(" ")).collect
res8: Array[String] = Array(coffee, panda, happy, panda, happiest, panda, party)
```

### reduceByKey(binary_function)
* reduceByKey就是对元素为KV对的RDD中Key相同的元素的Value进行binary_function的reduce操作，因此，Key相同的多个元素的值被reduce为一个值，然后与原RDD中的Key组成一个新的KV对。

```scala
val a = sc.parallelize(List((1,2),(1,3),(3,4),(3,6)))
a.reduceByKey((x,y) => x + y).collect
1
2
//结果 Array((1,5), (3,10))
```
### reduce(binary_function)
* reduce将RDD中元素前两个传给输入函数，产生一个新的return值，新产生的return值与RDD中下一个元素（第三个元素）组成两个元素，再被传给输入函数，直到最后只有一个值为止。
```scala
val c = sc.parallelize(1 to 10)
c.reduce((x, y) => x + y)//结果55
1
2
具体过程，RDD有1 2 3 4 5 6 7 8 9 10个元素，
1+2=3 3+3=6
6+4=10 10+5=15
15+6=21 21+7=28
28+8=36 36+9=45
```
## spark 分区相关
[读取文件流程](1.png)
### 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

* 输入可能以多个文件的形式存储在HDFS上，每个File都包含了很多块，称为Block。
当Spark读取这些文件作为输入时，会根据具体数据格式对应的InputFormat进行解析，一般是将若干个Block合并成一个输入分片，称为InputSplit，注意InputSplit不能跨越文件。
随后将为这些输入分片生成具体的Task。InputSplit与Task是一一对应的关系。
随后这些具体的Task每个都会被分配到集群上的某个节点的某个Executor去执行。

* 每个节点可以起一个或多个Executor。
* 每个Executor由若干core组成，每个Executor的每个core一次只能执行一个Task。
* 每个Task执行的结果就是生成了目标RDD的一个partiton。

* 参考链接： https://blog.csdn.net/weixin_38750084/article/details/82723130