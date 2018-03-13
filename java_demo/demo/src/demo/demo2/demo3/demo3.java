package demo.demo2.demo3;

import demo.demo.Animal;
import demo.demo2.demo2.Gun;

public class demo3 {
	public static void main(String args[]){
		Animal a = new Animal(5,"white","female","jack");
		a.eat(" grass");
		new Gun( 100).shot();
	}
}
