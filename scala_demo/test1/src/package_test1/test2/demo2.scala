package package_test1.test2
import package_test1.demo1.{animal_protected, Weapon}
//Weapon是定义在demo1中object之外的类
//animal_protected是定义在demo1 object之内的方法，导入进来只会可以直接使用，因为不是类所以不需要实例化

class Gun(price : Int , type: String, user:String) extends Weapon(price, type){
	var user_= user

	override def reload(num:Int):String={
		println("the gun is reloaded ")
    return "on the way"
	}
	override def fire(s:String):Unit={
		if(s == "on the way"){
      		println("the gun is working")
    	}else{
      		println("please reload the gun") 
    	}
	}
}


object demo2 {
  def main(args: Array[String]){
    animal_protected()
    demo1.animal_protected()//以上就是直接使用从demo1中导入进来的方法，直接使用
    
    val gun = new Gun(1000,"big gun","jack")
    gun.fire(gun.reload())
  }
}