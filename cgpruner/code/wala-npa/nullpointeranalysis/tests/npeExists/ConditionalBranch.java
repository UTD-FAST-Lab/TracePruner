package tests.npeExists;
public class ConditionalBranch {
  public ConditionalBranch y;

  public static void main(String[] args) {
    ConditionalBranch h1 = new ConditionalBranch();
    ConditionalBranch h2 = new ConditionalBranch();
    int x = 3;
    int z = 2;
    if (x < z*x){
    	h1 = h1.y;
      h2 = h2.y;
    }
    h1.foo(); // NPE
    h2.foo(); // NPE
  }

  public void foo(){
    int x;
  }
}
