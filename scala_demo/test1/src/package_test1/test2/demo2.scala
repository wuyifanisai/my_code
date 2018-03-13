package package_test1.test2
import package_test1.demo1.{animal_protected, Weapon}


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
    
    val gun = new Gun(1000,"big gun","jack")
    gun.fire(gun.reload())
  }
}