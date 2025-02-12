package nullpointeranalysis;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.wala.classLoader.IMethod;

// This data represents the value for the DataStructure.M array.
public class MethodInfo {
	// key = field signature
	HashMap<String,Def> thisInitialization;
	ArrayList<Val> args;
	// key = field signature
	HashMap<String,Def> post;
	Val returnVal;
	
	public MethodInfo(IMethod keyM) {
		thisInitialization = new HashMap<String,Def>();
		post = new HashMap<String,Def>();
		returnVal = new Val(keyM.getReference().getSignature() + "^Return-val");
		args = new ArrayList<Val>(keyM.getNumberOfParameters());
		for (int i = 0 ; i < keyM.getNumberOfParameters() ; i++) {
			args.add(new Val(keyM.getReference().getSignature() + "^arg[" + i + "]"));
		}
	}
}
