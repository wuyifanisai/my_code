package package_test1.test2.test3
import package_test1.demo1.animal_protected
import package_test1.test2.demo2.Gun

object demo3 {
  def main(args: Array[String]){
    animal_protected()
    
    val gun = new Gun(1000,"big gun","jack")
    gun.fire(gun.reload())
  }
}