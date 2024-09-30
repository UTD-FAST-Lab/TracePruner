package tests.noNPEexists;

public class MethodCallInit {
	MethodCallInit a1;

	public MethodCallInit(){
		foo3();
	}

	public static void main(String[] args) {
		MethodCallInit s = new MethodCallInit();
		s.a1.foo2();
	}

	public void foo2(){
		int x = 5;
	}

	public void foo3(){
		a1 = new MethodCallInit();
	}
}
