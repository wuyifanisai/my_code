/*
1.3.2 实例介绍
在本实例中将使用协同过滤算法对GroupLens Research（http://grouplens.org/datasets/movielens/）提供的数据进行分析，该数据为一组从20世纪90年末到21世纪初由MovieLens用户提供的电影评分数据，这些数据中包括电影评分、电影元数据（风格类型和年代）以及关于用户的人口统计学数据（年龄、邮编、性别和职业等）。根据不同需求该组织提供了不同大小的样本数据，不同样本信息中包含三种数据：评分、用户信息和电影信息。
对这些数据分析进行如下步骤：
1. 装载如下两种数据：
a)装载样本评分数据，其中最后一列时间戳除10的余数作为key，Rating为值；
b)装载电影目录对照表（电影ID->电影标题）
2.将样本评分表以key值切分成3个部分，分别用于训练 (60%，并加入用户评分), 校验 (20%), and 测试(20%)
3.训练不同参数下的模型，并再校验集中验证，获取最佳参数下的模型
4.用最佳模型预测测试集的评分，计算和实际评分之间的均方根误差
5.根据用户评分的数据，推荐前十部最感兴趣的电影（注意要剔除用户已经评分的电影）
1.3.3 测试数据说明
在MovieLens提供的电影评分数据分为三个表：评分、用户信息和电影信息，在该系列提供的附属数据提供大概6000位读者和100万个评分数据，具体位置为/data/class8/movielens/data目录下，对三个表数据说明可以参考该目录下README文档。
1.评分数据说明（ratings.data)
该评分数据总共四个字段，格式为UserID::MovieID::Rating::Timestamp，分为为用户编号：：电影编号：：评分：：评分时间戳，其中各个字段说明如下：
l用户编号范围1~6040
l电影编号1~3952
l电影评分为五星评分，范围0~5
l评分时间戳单位秒
l每个用户至少有20个电影评分
使用的ratings.dat的数据样本如下所示：
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
1::3408::4::978300275
1::2355::5::978824291
1::1197::3::978302268
1::1287::5::978302039
1::2804::5::978300719
2.用户信息(users.dat)
用户信息五个字段，格式为UserID::Gender::Age::Occupation::Zip-code，分为为用户编号：：性别：：年龄：：职业::邮编，其中各个字段说明如下：
l用户编号范围1~6040
l性别，其中M为男性，F为女性
l不同的数字代表不同的年龄范围，如：25代表25~34岁范围
l职业信息，在测试数据中提供了21中职业分类
l地区邮编
使用的users.dat的数据样本如下所示：
1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117
4::M::45::7::02460
5::M::25::20::55455
6::F::50::9::55117
7::M::35::1::06810
8::M::25::12::11413
3.电影信息(movies.dat)
电影数据分为三个字段，格式为MovieID::Title::Genres，分为为电影编号：：电影名：：电影类别，其中各个字段说明如下：
l电影编号1~3952
l由IMDB提供电影名称，其中包括电影上映年份
l电影分类，这里使用实际分类名非编号，如：Action、Crime等
使用的movies.dat的数据样本如下所示：
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
6::Heat (1995)::Action|Crime|Thriller
7::Sabrina (1995)::Comedy|Romance
8::Tom and Huck (1995)::Adventure|Children's
1.3.4 程序代码
*/

import java.io.File
import scala.io.Source
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}
 
object MovieLensALS {
 
  def main(args: Array[String]) {
    // 屏蔽不必要的日志显示在终端上
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
 
    if (args.length != 2) {
      println("Usage: /path/to/spark/bin/spark-submit --driver-memory 2g --class week7.MovieLensALS " +
        "week7.jar movieLensHomeDir personalRatingsFile")
      sys.exit(1)
    }
 
    // 设置运行环境
    val conf = new SparkConf().setAppName("MovieLensALS").setMaster("local[4]")
    val sc = new SparkContext(conf)
 
    // 装载用户评分，该评分由评分器生成
    val myRatings = loadRatings(args(1))
    val myRatingsRDD = sc.parallelize(myRatings, 1)
 
    // 样本数据目录
    val movieLensHomeDir = args(0)
 
    // 装载样本评分数据，其中最后一列Timestamp取除10的余数作为key，Rating为值,即(Int,Rating)
    val ratings = sc.textFile(new File(movieLensHomeDir, "ratings.dat").toString).map { line =>
      val fields = line.split("::")
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }
 
    // 装载电影目录对照表（电影ID->电影标题）
    val movies = sc.textFile(new File(movieLensHomeDir, "movies.dat").toString).map { line =>
      val fields = line.split("::")
      (fields(0).toInt, fields(1))
    }.collect().toMap
 
    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct().count()
    val numMovies = ratings.map(_._2.product).distinct().count()
 
    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")
 
    // 将样本评分表以key值切分成3个部分，分别用于训练 (60%，并加入用户评分), 校验 (20%), and 测试(20%)
    // 该数据在计算过程中要多次应用到，所以cache到内存
    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD) //注意ratings是(Int,Rating)，取value即可
      .repartition(numPartitions)
      .cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()
    val test = ratings.filter(x => x._1 >= 8).values.cache()
 
    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()
 
    println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest)
 
    // 训练不同参数下的模型，并在校验集中验证，获取最佳参数下的模型
    val ranks = List(8, 12)
    val lambdas = List(0.1, 10.0)
    val numIters = List(10, 20)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRmse(model, validation, numValidation)
      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }
 
    // 用最佳模型预测测试集的评分，并计算和实际评分之间的均方根误差
    val testRmse = computeRmse(bestModel.get, test, numTest)
 
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda  + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")
 
    // create a naive baseline and compare it with the best model
    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRmse =
      math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
    val improvement = (baselineRmse - testRmse) / baselineRmse * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")
 
    // 推荐前十部最感兴趣的电影，注意要剔除用户已经评分的电影
    val myRatedMovieIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel.get
      .predict(candidates.map((0, _)))
      .collect()
      .sortBy(-_.rating)
      .take(10)
 
    var i = 1
    println("Movies recommended for you:")
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product))
      i += 1
    }
 
  sc.stop()
  }
 
  /** 校验集预测数据和实际数据之间的均方根误差 **/
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }
 
  /** 装载用户评分文件 **/
  def loadRatings(path: String): Seq[Rating] = {
    val lines = Source.fromFile(path).getLines()
    val ratings = lines.map { line =>
      val fields = line.split("::")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.filter(_.rating > 0.0)
    if (ratings.isEmpty) {
      sys.error("No ratings provided.")
    } else {
      ratings.toSeq
    }
  }
}