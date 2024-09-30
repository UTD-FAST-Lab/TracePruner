package nullpointeranalysis;

import java.util.HashMap;

import com.ibm.wala.classLoader.IMethod;

public class DataStructure {
	// Key = field-signature, Value = (whether null or not)
	public static HashMap<String,Val> H;
	
	// Key = <method-signature>#<variable-value-number>, Value = (whether null or not)
	public static HashMap<String,Val> L;
	
	// Key = <method-signature>#<variable-value-number>#<field-signature>, Value = (whether null or not)
	public static HashMap<String,Def> T;
		
	// Key = <method-signature>, Value = Method-info object with the 4 fields.
	public static HashMap<String, MethodInfo> M;
	
	// Initializing all the maps.
	static {
		H = new HashMap<String,Val>();
		L = new HashMap<String,Val>();
		T = new HashMap<String,Def>();
		M = new HashMap<String, MethodInfo>();
	}
}

class Pair{
	IMethod first;
	IMethod second;
	public Pair(IMethod m1, IMethod m2) {
		first = m1;
		second = m2;
	}
}