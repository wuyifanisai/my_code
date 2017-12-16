import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf , SparkContext}
import org.apache.spark.rdd.RDD

object RunWordCount {
  def main( args:  Array[String]) : Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    System.setProperty("spark.ui.show.ConsoleProgress", "false")
    
    println("begin running RunWordCount...")
    val sc = new SparkContext(new SparkConf().setAppName("wordcount").setMaster("local[2]"))
    
    println("begin read the file...")
    val textFile = sc.textFile("data/LICENSE.txt")
    
    println("begin create RDD...")
    val countsRDD = textFile.flatMap(line=>line.split(" ")).map(word=>(word,1)).reduceByKey(_+_)
    
    println("begin save the result...")
    try {
      countsRDD.saveAsTextFile("data/output")
      println("saved !")
    }catch{
      case e: Exception =>println("remove the file before")
    }
  }
}