package tests.noNPEexists;

public class InitializedInOtherFunction {
	InitializedInOtherFunction f;

	public InitializedInOtherFunction(){
		foo3();
	}

	public static void main(String[] args) {
		InitializedInOtherFunction s = new InitializedInOtherFunction();
		s.f.foo2();
	}

	public void foo2(){
		int x = 0;
	}

	public void foo3(){
		f = new InitializedInOtherFunction();
	}
}
