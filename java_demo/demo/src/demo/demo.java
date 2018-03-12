/*
 * 这是一个java的demo
 * 其中简单包括了public class的定义
 * public class中主方法的定义
 * 主方法中类的定义以及相应的主要执行内容（其中包括了类的实例化以及向上向下转型的过程）
 * 在主方法外定义了static class
 * 以及子类与父类
 * 以及相应接口的定义与使用
 * 以及类型的判断
 */

package demo;

public class demo {
	
	public static void main(String args[]){
		
		class People{//因为是在主方法之内定义的，所以不用需要static关键词，以便主方法中可调用
			int age;
			String name;
			public People(int age, String name){ //构造方法
				this.age = age;
				this.name = name;
			}
			public void hunting(Animal animal){ //类中的普通无输出方法
				System.out.println("a human named "+this.name+" is hunting a "+animal.name);
			}
		}
		
		//main 中的first主要执行内容
		Dog snoopy = new Dog(3,"yellow","female","dog");
		Cat amy = new Cat(2,"white","female","cat");
		snoopy.move();
		amy.eat("fish");
		amy.live(new Live_cat());
		snoopy.live(new Live_dog());
		System.out.println("now human is doing something bad...");
		People people = new People(33, "Tom");
		people.hunting(new Animal(4,"white","female","jack"));
		animal_protected();//调用定义在在主方法外部的方法
		
		//main中的second主要执行内容
		Animal animal = new Dog(10,"red","female","dog_red"); //向上转型
		animal.live(new Live_dog()); //调用dog类型里面的live方法
		
		Dog d = (Dog) animal; //向下转型
		d.move(); //调用dog的move方法
		show(d);//定义在主方法外部的类型判断函数
	
	}


	static class Animal { //因为是在主方法之外定义的，所以需要static关键词，以便主方法中可调用
		int age;
		String color;
		String sex;
		String name;
		
		public Animal(int age, String color, String sex, String name){
			this.age = age;
			this.color = color;
			this.sex = sex;
			this.name = name;
		}
		
		public void move(){
			System.out.println("this anmial is runing...");
		}
		
		public void eat(String food){
			System.out.println("this anmial is eating" + food);
		}
		
		public void live(Anmial_generate ag){ //调用接口
			ag.mating();
			ag.delivering();
		}
	}
	
	static class Dog extends Animal{//因为是在主方法之外定义的，所以需要static关键词，以便主方法中可调用
		
		public Dog(int age, String color, String sex, String name){
			super(age,color,sex, name);
		}
		
		public void move(){//复写父类的方法
			System.out.println("this dog is runing...");
		}
		
		public void eat(String food){//复写父类的方法
			System.out.println("this dog is eating" + food);
		}
	}
	
	
	static class Cat extends Animal{//因为是在主方法之外定义的，所以需要static关键词，以便主方法中可调用
		public Cat(int age, String color, String sex, String name){
			super(age, color, sex,name);
		}
		
		public void move(){//复写父类的方法
			System.out.println("this cat is moving...");
		}
		
		public void eat(String food){//复写父类的方法
			System.out.println("this cat is eating"+food);
		}
		
		
	}
	
	interface Anmial_generate {//定义一个接口
		public void mating();
		public void delivering();
	}
	
	static class Live_dog implements Anmial_generate{//因为是在主方法之外定义的，所以需要static关键词，以便主方法中可调用
		//定义具体的接口实现类
		@Override
		public void mating() {
			System.out.println("the dog is mating");// TODO Auto-generated method stub
		}

		@Override
		public void delivering() {
			System.out.println("the dog is delivering ...");// TODO Auto-generated method stub
			
		}
		
	}
	
	static class Live_cat implements Anmial_generate{
		//定义具体的接口实现类
		@Override
		public void mating() {
			System.out.println("the cat is mating");// TODO Auto-generated method stub
		}

		@Override
		public void delivering() {
			System.out.println("the cat is delivering ...");// TODO Auto-generated method stub
			
		}
		
	}
	
	static void animal_protected(){ //因为是在主方法之外定义的，所以需要static关键词，以便主方法中可调用
		System.out.println("human shuold not kill wild animal, wild animal should be protected !");
	}
	
	public static void show(Animal a){
		//类型判断
		if (a instanceof Dog){
			Dog d = (Dog) a;
			d.live(new Live_dog());
		}
		else {
			Cat c = (Cat) a;
			c.live(new Live_cat());
		}
	}
	
}
