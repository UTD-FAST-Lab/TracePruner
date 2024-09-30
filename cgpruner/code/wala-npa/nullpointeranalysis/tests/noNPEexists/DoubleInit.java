package tests.noNPEexists;


public class DoubleInit {
	DoubleInit a1;

	public DoubleInit(DoubleInit d) {
		a1 = d;
	}
	public DoubleInit(){
		this(new DoubleInit());
	}

	public static void main(String[] args) {
		DoubleInit s = new DoubleInit();
		s.a1.foo();
	}

	public void foo(){
		int x = 5;
	}
}
