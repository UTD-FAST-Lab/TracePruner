package tests.npeExists;

public class X {
	X a1;

	public X(){
		foo3();
	}

	public static void main(String[] args) {
		X s = new X();
		s.a1.foo2();
	}

	public void foo2(){
		int x = 5;
	}

	public void foo3(){
		int x = 4;
	}
}
