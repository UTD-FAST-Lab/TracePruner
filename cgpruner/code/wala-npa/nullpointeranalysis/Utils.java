package nullpointeranalysis;

import java.util.HashMap;

import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.types.FieldReference;

public class Utils {
	// Creates a key for DataStructure.T
	public static String makeTKey(IMethod m, int lineNum, FieldReference f) {
		String key = m.getSignature() + "#" + lineNum + "#" + f.getSignature();
		return key;
	}
	
	// Creates a key for DataStructure.L
	public static String makeLKey(IMethod m, int varValueNumber) {
		String key = m.getSignature() + "#" + varValueNumber;
		return key;
	}
	
	// Creates an empty entry for keyL in map if it doesn't exist,
	// and then return the corresponding value from the map
	public static Val getOrCreateVal(HashMap<String, Val> map, String keyL) {
		if (!map.containsKey(keyL)) {
			map.put(keyL, new Val(keyL));
		}
		return map.get(keyL);
	}
	
	// Returns value of keyT (Creates entry if it doesn't exist)
	public static Def getOrCreateDef(HashMap<String, Def> map, String keyT) {
		if (!map.containsKey(keyT)) {
			map.put(keyT, new Def(keyT));
		}
		return map.get(keyT);
	}

	// Returns value of keyM (Creates entry if it doesn't exist)
	public static MethodInfo getOrCreateMethodInfo(HashMap<String, MethodInfo> map, IMethod keyM) {
		String mSignature = keyM.getReference().getSignature();
		if (!map.containsKey(mSignature)) {
			map.put(mSignature, new MethodInfo(keyM));
		}
		return map.get(mSignature);
	}	
	
	// Combines the calls to make the key and get the map
	public static Def getTElement(IMethod currentMethod, int line, FieldReference f) {
		String keyT1 = makeTKey(currentMethod, line, f);
		Def defElement = Utils.getOrCreateDef(DataStructure.T, keyT1);
		return defElement;
	}
	
	// Combines the calls to make the key and get the map
	public static Val getLElement(IMethod currentMethod, int valueNumber) {
		String keyL1 = makeLKey(currentMethod, valueNumber);
		Val valElement = Utils.getOrCreateVal(DataStructure.L, keyL1);
		return valElement;
	}
}
