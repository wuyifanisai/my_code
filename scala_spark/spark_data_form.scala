import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.log4j.{Logger}
import org.apache.log4j.Level

object spark_data_form {
  def main(args: Array[String]): Unit = {
     SetLogger()
   
    val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[8]"))
    val sqlContext = new SQLContext(sc)
    
    //try to see some usual data type in spark RDD and dataframe
    val data = Array(  // Vectors is only one feature
      Vectors.sparse(5, Seq((1,1.0), (3,7.0))),
      Vectors.dense(2,0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
     )
     val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("f1")
    println("df")
    df.show()
    println()
    
    val df2 = Seq(   
                           (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
                           (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
                           (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
                          )
    val df3 = sqlContext.createDataFrame(df2).toDF("id", "features", "clicked")
    println("df3")
    df3.rdd.foreach(x => println(x))
    println()
    
    val df4 = sqlContext.createDataFrame(Seq(
                                                                (0, "a"),
                                                                (1, "b"),
                                                                (2, "c"),
                                                                (3, "a"),
                                                                (4, "a"),
                                                                (5, "c")
                                                                )).toDF("id", "category")
    println("df4")
    df4.rdd.foreach(x => println(x))
    println()
    
    val df5 = Array((0, 0.1,3), (1, 0.8,5), (2, 0.2,8))  // there are one label and two feature in a array
    val df6 = sqlContext.createDataFrame(df5).toDF("id", "f1","f2")
    df6.show()
    println()

     //Create a dense vector (1.0, 0.0, 3.0).
      val v = Vectors.dense(1.0, 5.0, 3.0,9.0)
      println("normal vector:")
      println(v)
      
      // Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries
      val v1 = Vectors.sparse(4,Array(0,1), Array(1,1))
      println("sparse vector:")
      println(v1)
      
      // Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries
      val v2 = Vectors.sparse(4,Seq((1, 1.0), (3, 7.0)))
      println("sparse vector:")
      println(v2)
    
      // creart a labelpoint vector
      val v4 = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
      val v5 = LabeledPoint(1.0, Vectors.sparse(4,Array(0,1), Array(2,3)))
      println("labeledpoint vector:")
      println(v4)
      println(v5)
      
      // matrix in local position
     val m = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0) )
     println("local matrix:")
     println(m)
     println()
     
     
      /*
     实际运用中,稀疏数据是很常见的。MLlib可以读取以LIBSVM格式存储的训练实例,LIBSVM格式是 LIBSVM 和 LIBLINEAR的默认格式,这是一种文本格式,每行代表一个含类标签的稀疏特征向量。格式如下:
			label index1:value1 index2:value2 ...
     索引是从 1 开始并且递增。加载完成后,索引被转换为从 0 开始。
     通过 MLUtils.loadLibSVMFile读取训练实例并以LIBSVM 格式存储。
			Plain Text code

			import org.apache.spark.mllib.regression.LabeledPoint
			import org.apache.spark.mllib.util.MLUtils
			import org.apache.spark.rdd.RDD
			val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
			*/
     
     // read from file to get RDD data
     import org.apache.spark.rdd.RDD
     import org.apache.spark.mllib.util.MLUtils
     val rdd_data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/wuyifanhadoop/test/sample_libsvm_data.txt")
     rdd_data.foreach { x => println(x) }
     
     val df_data = sqlContext.createDataFrame(rdd_data).toDF("label","features")
     df_data.show()
   

  }
  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}