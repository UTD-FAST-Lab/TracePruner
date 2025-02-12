package tests.noNPEexists;

public class StaticMethodTest {
	StaticMethodTest y;
	public static void main(String[] args) {
		StaticMethodTest s = new StaticMethodTest();
		foo(s);
	}
	
	public static void foo(StaticMethodTest s) {
 		s = new StaticMethodTest();
	}
}
