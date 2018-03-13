package package_test1

/*
 * it is a demo of scala code
 * it is about main functin, class, subclass,extends,trait, 
 * 
 * 
 * 
 */
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Normalizer, Binarizer, Bucketizer}
// ML Feature Creation

import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
//import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
// there is no TrainValidationSplit ??
// tuning hyper parameters

import org.apache.spark.ml.evaluation.{RegressionEvaluator}
// evaluation for regression model

import org.apache.spark.ml.regression.{LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor}
// regression model
// we will use some different models and compare them with each other ,finally a best model with best parameters would be appeared ! 

import org.apache.spark.ml.Pipeline
// pipeline
import org.apache.spark.rdd.RDD


trait live{//定义了一个借口， 用于规范animal的live操作
  def mating():Unit
  def generating():Unit
}

class Weapon(price : Int, type : String){
  var price_ = price
  val type_ = type

  def reload(num:Int): String = {
    println("the Weapon is reloaded ")
    return "on the way"
  }

  def fire(s:String):Unit={
    if(s == "on the way"){
      println("the Weapon is working")
    }else{
      println("please reload") 
    }
  }

}

class Animal( age: Int ,  sex:String, color:String) extends live {
  //定义了一个animal的类，同时接上了live的借口
  var age_ : Int = age;//年龄是会变的，所以用var
  val sex_ : String = sex;
  val color_ :String = color;
  
  def move(speed : Int, direction:String):Unit ={//定义了一个无返回值得方法
    println("the animal is move at a speed of %d and the direction is %s", speed , direction );
  }
  def isOld():String = {//定义了一个有返回值的方法
    if (age > 18){
      val s = "true";
      return s;
    }else{
      val s = "false";
      return s;
    }
  }
  
  def mating():Unit ={//重写接口中的方法
    println("the animal is mating")
  }
  def generating():Unit={//重写接口中的方法
    println("the animal is generating")
  }
}

class Dog(age : Int , sex:String, color :String, name:String) extends Animal(age, sex, color){
  //定义了一个dog类，继承animal的类
  val name_ = name;
  override def move(speed : Int, direction:String):Unit ={//重写父类的方法
    super.move(speed, direction);//可以调用一下父类的方法
    println("the dog is move at a speed of %d and the direction is %s", speed , direction );
  }
  def eat():Unit={//定义一个父类中没有的方法
    println("the dog is eating...")
  }
}


object demo1 {
  
  def animal_protected():Unit={//在主方法中定义一个方法
  println("we should protect wild animal")
  }
  
  def main(args: Array[String]) {//主方法，程序的入口
    println("the main function begins...")
    
    val  animal = new Animal(5,"female","white");//类的实例化
    animal.move(10, "west");
    animal.isOld();
    println(animal.age_)
    animal.age_ = 10;
    println(animal.age_)
    animal.mating()
    
    val dog = new Dog(3,"male","black","snoopy");
    println(dog.color_);
    println(dog.name_);
    dog.move(20, "east")
    dog.eat()
    println(dog.isOld())
    dog.generating()
    
    new Dog(5, "female","white","amy").eat();//类似于匿名类的实例化对象以及调用方法
    
    animal_protected()
   
    def evaluateModel(model:LogisticRegressionModel, validationData:RDD[LabeledPoint]):Double={
      //在住方法中定义的方法
      val predict_labels = validationData.map{
        data =>
        var predict = model.predict(data.features)
        (predict, data.label)
      }
      val Metrics = new BinaryClassificationMetrics(predict_labels)
      val auc = Metrics.areaUnderROC
      return auc
    }
    println("the main function is over")
    
}
}