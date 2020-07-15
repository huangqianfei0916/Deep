## scala基础语法

### 变量与常量
* 变量用var，常量用val


### 方法与函数
* Scala 中的方法跟 Java 的类似，方法是组成类的一部分。

* Scala 中的函数则是一个完整的对象，Scala 中的函数其实就是继承了 Trait 的类的对象。

* Scala 中使用 val 语句可以定义函数，def 语句定义方法。

* 方法的定义

```scala
def functionName ([参数列表]) : [return type] = {
   function body
   return [expr]
}
```
* 1 函数可以作为一个参数传递给方法；
```scala
object MethodAndFunctionDemo {
  //定义一个方法
  //方法 m1 参数要求是一个函数，函数的参数必须是两个Int类型
  //返回值类型也是Int类型
  def m1(f:(Int,Int) => Int) : Int = {
    f(2,6)
  }

  //定义一个函数f1,参数是两个Int类型，返回值是一个Int类型
  val f1 = (x:Int,y:Int) => x + y
  //再定义一个函数f2
  val f2 = (m:Int,n:Int) => m * n

  def main(args: Array[String]): Unit = {
    //调用m1方法，并传入f1函数
    val r1 = m1(f1)

    println(r1)

    //调用m1方法，并传入f2函数
    val r2 = m1(f2)
    println(r2)
  }
}
```
* 2 在Scala中无法直接操作方法，如果要操作方法，必须先将其转换成函数。有两种方法可以将方法转换成函数：
```
val f1 = m _
```
```scala
object TestMap {

  def ttt(f:Int => Int):Unit = {
    val r = f(10)
    println(r)
  }

  val f0 = (x : Int) => x * x

  //定义了一个方法
  def m0(x:Int) : Int = {
    //传递进来的参数乘以10
    x * 10
  }
  //将方法转换成函数，利用了神奇的下滑线
  val f1 = m0 _

  def main(args: Array[String]): Unit = {
    ttt(f0)
    //通过m0 _将方法转化成函数
    ttt(m0 _);
    //如果直接传递的是方法名称，scala相当于是把方法转成了函数
    ttt(m0)
    //通过x => m0(x)的方式将方法转化成函数,这个函数是一个匿名函数，等价：(x:Int) => m0(x)
    ttt(x => m0(x))
  }
}
```
* 3 函数必须要有参数列表，而方法可以没有参数列表
```
def m1=100; 正确
val f1=（）=>100;正确
val f1= =>100；错误
````
* 数组

* 数组的声明
* var z:Array[String] = new Array[String](3)
* 数组的初始化和遍历
```scala
var myList = Array(1.9, 2.9, 3.4, 3.5)
      
      // 输出所有数组元素
      for ( x <- myList ) {
         println( x )
      }

      // 计算数组所有元素的总和
      var total = 0.0;
      for ( i <- 0 to (myList.length - 1)) {
         total += myList(i);
      }
      println("总和为 " + total);
```
* List Set Map 元组
```scala
// 定义整型 List
val x = List(1,2,3,4)

// 定义 Set
val x = Set(1,3,5,7)

// 定义 Map
val x = Map("one" -> 1, "two" -> 2, "three" -> 3)

// 创建两个不同类型元素的元组
val x = (10, "Runoob")
```