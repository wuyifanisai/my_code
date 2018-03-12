package demo;

public class demo {
	
	public static void main(String args[]){
		
		class People{
			int age;
			String name;
			public People(int age, String name){
				this.age = age;
				this.name = name;
			}
			public void hunting(Animal animal){
				System.out.println("a human named "+this.name+" is hunting a "+animal.name);
			}
		}
	
		Dog snoopy = new Dog(3,"yellow","female","dog");
		Cat amy = new Cat(2,"white","female","cat");
		snoopy.move();
		amy.eat("fish");
		amy.live(new Live_cat());
		snoopy.live(new Live_dog());
		System.out.println("now human is doing something bad...");
		People people = new People(33, "Tom");
		people.hunting(new Animal(4,"white","female","jack"));
	
	}
	
	static class Animal {
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
		
		public void live(Anmial_generate ag){
			ag.mating();
			ag.delivering();
		}
	}
	
	static class Dog extends Animal{
		
		public Dog(int age, String color, String sex, String name){
			super(age,color,sex, name);
		}
		
		public void move(){
			System.out.println("this dog is runing...");
		}
		
		public void eat(String food){
			System.out.println("this dog is eating" + food);
		}
	}
	
	
	static class Cat extends Animal{
		public Cat(int age, String color, String sex, String name){
			super(age, color, sex,name);
		}
		
		public void move(){
			System.out.println("this cat is moving...");
		}
		
		public void eat(String food){
			System.out.println("this cat is eating"+food);
		}
		
		
	}
	
	interface Anmial_generate {
		public void mating();
		public void delivering();
	}
	
	static class Live_dog implements Anmial_generate{

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

		@Override
		public void mating() {
			System.out.println("the cat is mating");// TODO Auto-generated method stub
		}

		@Override
		public void delivering() {
			System.out.println("the cat is delivering ...");// TODO Auto-generated method stub
			
		}
		
	}
	
}
