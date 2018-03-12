//其他java文件中的类可以导入，但是定义在那个jave文件中单独定义的方法无法导入其中
package demo.demo2;
import demo.demo.Animal;
import demo.demo.Anmial_generate;
import demo.demo.Cat;
import demo.demo.Dog;
import demo.demo.Live_cat;
import demo.demo.Live_dog;
//import demo.demo.show;

public class demo2 {
	public static void main(String args[]) {

		class People<T> {// 因为是在主方法之内定义的，所以不用需要static关键词，以便主方法中可调用
			// 定义了一个泛型类
			int age;
			String name;
			private T t;// 定义一个泛型

			public People(int age, String name) { // 构造方法
				this.age = age;
				this.name = name;
			}

			public void add(T t) {
				this.t = t;
			}

			public T get() {
				return this.t;
			}

			public void hunting(Animal animal) { // 类中的普通无输出方法
				System.out.println("a human named " + this.name + " is hunting a " + animal.name);
			}

			public <E> void fun(E[] array) { // 定义一个泛型方法
				for (E element : array) {
					System.out.println(element);
				}
			}

		}

		// main 中的first主要执行内容
		Dog snoopy = new Dog(3, "yellow", "female", "dog");
		Cat amy = new Cat(2, "white", "female", "cat");
		snoopy.move();
		amy.eat("fish");
		amy.live(new Live_cat());
		snoopy.live(new Live_dog());

		System.out.println("now human is doing something bad...");
		People people = new People<String>(33, "Tom");
		people.add("this man is a bad guy!");
		System.out.println(people.get());

		String[] s = { "black man ", "white man ", "yellow man " };
		people.fun(s);

		people.hunting(new Animal(4, "white", "female", "jack"));
		//animal_protected();// 调用定义在在主方法外部的方法

		// main中的second主要执行内容
		Animal animal = new Dog(10, "red", "female", "dog_red"); // 向上转型
		animal.live(new Live_dog()); // 调用dog类型里面的live方法

		Dog d = (Dog) animal; // 向下转型
		d.move(); // 调用dog的move方法
		//show(d);// 定义在主方法外部的类型判断函数

	}


	
}
