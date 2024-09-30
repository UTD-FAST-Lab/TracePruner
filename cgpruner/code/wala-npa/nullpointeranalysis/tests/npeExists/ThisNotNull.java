package tests.npeExists;

// This tries to check whether the analysis correctly
// implements the constraint of 'this' never being "maybeNull".
// So this case should return just 1 NPE
public class ThisNotNull {
  public ThisNotNull y;

  public static void main(String[] args) {
    ThisNotNull h = new ThisNotNull();
    ThisNotNull s = h.y;
    ThisNotNull z = s.foo(); // should be flagged as NPE
  }

  public ThisNotNull foo(){
    return this.y; // should not be flagged as an NPE.
  }
}
