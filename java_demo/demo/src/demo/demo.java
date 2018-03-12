/*
 * ����һ��java��demo
 * ���м򵥰�����public class�Ķ���
 * public class���������Ķ���
 * ����������Ķ����Լ���Ӧ����Ҫִ�����ݣ����а��������ʵ�����Լ���������ת�͵Ĺ��̣�
 * ���������ⶨ����static class
 * �Լ������븸��
 * �Լ���Ӧ�ӿڵĶ�����ʹ��
 * �Լ����͵��ж�
 */

package demo;

public class demo {

	public static void main(String args[]) {

		class People<T> {// ��Ϊ����������֮�ڶ���ģ����Բ�����Ҫstatic�ؼ��ʣ��Ա��������пɵ���
			// ������һ��������
			int age;
			String name;
			private T t;// ����һ������

			public People(int age, String name) { // ���췽��
				this.age = age;
				this.name = name;
			}

			public void add(T t) {
				this.t = t;
			}

			public T get() {
				return this.t;
			}

			public void hunting(Animal animal) { // ���е���ͨ���������
				System.out.println("a human named " + this.name + " is hunting a " + animal.name);
			}

			public <E> void fun(E[] array) { // ����һ�����ͷ���
				for (E element : array) {
					System.out.println(element);
				}
			}

		}

		// main �е�first��Ҫִ������
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
		animal_protected();// ���ö��������������ⲿ�ķ���

		// main�е�second��Ҫִ������
		Animal animal = new Dog(10, "red", "female", "dog_red"); // ����ת��
		animal.live(new Live_dog()); // ����dog���������live����

		Dog d = (Dog) animal; // ����ת��
		d.move(); // ����dog��move����
		show(d);// �������������ⲿ�������жϺ���

	}

	public static class Animal { // ��Ϊ����������֮�ⶨ��ģ�������Ҫstatic�ؼ��ʣ��Ա��������пɵ���
		int age;
		String color;
		String sex;
		public String name;

		public Animal(int age, String color, String sex, String name) {
			this.age = age;
			this.color = color;
			this.sex = sex;
			this.name = name;
		}

		public void move() {
			System.out.println("this anmial is runing...");
		}

		public void eat(String food) {
			System.out.println("this anmial is eating" + food);
		}

		public void live(Anmial_generate ag) { // ���ýӿ�
			ag.mating();
			ag.delivering();
		}
	}

	public static class Dog extends Animal {// ��Ϊ����������֮�ⶨ��ģ�������Ҫstatic�ؼ��ʣ��Ա��������пɵ���

		public Dog(int age, String color, String sex, String name) {
			super(age, color, sex, name);
		}

		public void move() {// ��д����ķ���
			System.out.println("this dog is runing...");
		}

		public void eat(String food) {// ��д����ķ���
			System.out.println("this dog is eating" + food);
		}
	}

	public static class Cat extends Animal {// ��Ϊ����������֮�ⶨ��ģ�������Ҫstatic�ؼ��ʣ��Ա��������пɵ���
		public Cat(int age, String color, String sex, String name) {
			super(age, color, sex, name);
		}

		public void move() {// ��д����ķ���
			System.out.println("this cat is moving...");
		}

		public void eat(String food) {// ��д����ķ���
			System.out.println("this cat is eating" + food);
		}

	}

	public interface Anmial_generate {// ����һ���ӿ�
		public void mating();

		public void delivering();
	}

	public static class Live_dog implements Anmial_generate {// ��Ϊ����������֮�ⶨ��ģ�������Ҫstatic�ؼ��ʣ��Ա��������пɵ���
		// �������Ľӿ�ʵ����
		@Override
		public void mating() {
			System.out.println("the dog is mating");// TODO Auto-generated
													// method stub
		}

		@Override
		public void delivering() {
			System.out.println("the dog is delivering ...");// TODO
															// Auto-generated
															// method stub

		}

	}

	public static class Live_cat implements Anmial_generate {
		// �������Ľӿ�ʵ����
		@Override
		public void mating() {
			System.out.println("the cat is mating");// TODO Auto-generated
													// method stub
		}

		@Override
		public void delivering() {
			System.out.println("the cat is delivering ...");// TODO
															// Auto-generated
															// method stub

		}

	}

	public static void animal_protected() { // ��Ϊ����������֮�ⶨ��ģ�������Ҫstatic�ؼ��ʣ��Ա��������пɵ���
		System.out.println("human shuold not kill wild animal, wild animal should be protected !");
	}

	public static void show(Animal a) {
		// �����ж�
		if (a instanceof Dog) {
			Dog d = (Dog) a;
			d.live(new Live_dog());
		} else {
			Cat c = (Cat) a;
			c.live(new Live_cat());
		}
	}

}