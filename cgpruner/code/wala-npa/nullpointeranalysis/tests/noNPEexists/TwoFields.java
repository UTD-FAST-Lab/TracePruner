package tests.noNPEexists;

public class TwoFields {
	TwoFields a1;
	TwoFields a2;

	public TwoFields(){
		a1 = new TwoFields();
	}

	public static void main(String[] args) {
		TwoFields s = new TwoFields();
		foo(s.a1);
	}
	
	public static void foo(TwoFields smt) {
 		smt.foo2();
	}

	public void foo2(){
		int x = 5;
	}
}
