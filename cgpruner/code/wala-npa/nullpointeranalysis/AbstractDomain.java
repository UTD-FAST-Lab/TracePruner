package nullpointeranalysis;

import com.ibm.wala.classLoader.IClass;

public class AbstractDomain {
	String element; // The element that this abstract value represents.
}

//AbstractDomain corresponding to Val#
class Val extends AbstractDomain implements Comparable<Val>{
	ValBase baseRepresentation; 
	// rawClass is the class X in Raw(X). Hence only used if baseRepresentation==RAW
	IClass rawClass;  
	
	public Val(String e) {
		baseRepresentation = ValBase.NOTNULL;
		rawClass = null; 
		element = e;
	}
	public Val(ValBase b, IClass s, String e) {
		baseRepresentation = b;
		rawClass = s; 
		element = e;
	}
	@Override
	public int compareTo(Val o) {
		if (this.baseRepresentation == o.baseRepresentation) {
			if (this.rawClass==o.rawClass) {
				return 0;
			}
			else if (WalaNullPointerAnalysis.cha.isSubclassOf(this.rawClass, o.rawClass)) {
				return -1;
			} else {
				return 1;
			} 
		} else {
			return this.baseRepresentation.compareTo(o.baseRepresentation);
		}
	}	
	
	@Override
	public boolean equals(Object o) {
		Val v = (Val) o;
		if (this.baseRepresentation == v.baseRepresentation) {
			if (this.baseRepresentation == ValBase.RAW) {
				if(this.rawClass.equals(v.rawClass)) {
					return true;
				} else {
					return false;
				}
			} else {
				return true;
			}
		} else {
			return false;
		}
	}
}
enum ValBase{
	NOTNULL, RAW, RAWT, MAYBENULL
}

//AbstractDomain corresponding to Def#
class Def extends AbstractDomain implements Comparable<Def>{
	DefBase base;
	public Def(String e) {
		base = DefBase.DEF;
		element = e;
	}
	public Def(DefBase d, String e) {
		base = d;
		element = e;
	}
	@Override
	public int compareTo(Def o) {
		return this.base.compareTo(o.base);
	}
	
	@Override
	public boolean equals(Object o) {
		Def d = (Def) o;
		return this.base.equals(d.base);
	}
}
enum DefBase{
	DEF, UNDEF
}