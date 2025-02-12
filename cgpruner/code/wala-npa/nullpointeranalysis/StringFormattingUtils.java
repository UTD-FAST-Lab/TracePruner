package nullpointeranalysis;

import java.util.Arrays;

import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.types.Selector;
import com.ibm.wala.types.TypeName;

// Almost all the code in this file is copied from the WALA call-graph code.
public class StringFormattingUtils {
	// Lambda start string constants to identify lambda methods
	public static String walaLambdaStartString = "wala/lambda$";
	public static String lambdaMetafactoryClass = "java/lang/invoke/LambdaMetafactory";
	public static String walaArrayCopy = "com/ibm/wala/model/java/lang/System.arraycopy:(Ljava/lang/Object;Ljava/lang/Object;)V";
	public static String javaLibArrayCopy = "java/lang/System.arraycopy:(Ljava/lang/Object;ILjava/lang/Object;II)V";

	// Reformat the method if it is a lambda. Else simply return it.
	public static String reformatIfLambda(String inputMethod) {
		String outputMethod;
		if (inputMethod.startsWith(walaLambdaStartString)) {
			String fullLambdaSignature = inputMethod.substring(walaLambdaStartString.length()); // remove wala start
																								// string
			String lambdaSignatureWithoutArgs = fullLambdaSignature.split(":")[0];

			String classname = lambdaSignatureWithoutArgs.split("\\.")[0];
			String[] classnameListFormat = classname.split("\\$");
			String lambdaIndex = classnameListFormat[classnameListFormat.length - 1];
			// remove the last element (the lambda index) from the classname
			classnameListFormat = Arrays.copyOf(classnameListFormat, classnameListFormat.length - 1);
			String classnameFormatted = String.join("/", classnameListFormat);

			String methodname = lambdaSignatureWithoutArgs.split("\\.")[1];
			outputMethod = classnameFormatted + ".<lambda/" + methodname + "$" + lambdaIndex + ">:()V";
			return outputMethod;
		} else { // If it is not a lambda method
			return inputMethod;
		}
	}

	// format the method to the required bytecode format
	public static String formatMethod(TypeName t, String methodname, Selector sel) {
		String qualifiedClassName = "" + (t.getPackage() == null ? "" : t.getPackage() + "/") + t.getClassName();
		if (qualifiedClassName.equals(lambdaMetafactoryClass)) {
			return null; // don't want to use lambda metafactory nodes. They go nowhere, and don't appear
							// in javaq
		}
		String formattedMethod = qualifiedClassName + "." + methodname + ":" + sel.getDescriptor();
		// Modify the method if it is a lambda
		formattedMethod = reformatIfLambda(formattedMethod);
		// If it is wala arrayCopy, replace with java Arraycopy
		if (formattedMethod.equals(walaArrayCopy)) {
			formattedMethod = javaLibArrayCopy;
		}
		return formattedMethod;
	}

	// formats the final output line
	public static String formatFinalOutput(String firstMethod, String secondMethod, boolean bootSrcMethod, int off) {
		// Decide the bytecode offset (and fix firstMethod) depending on if it is a boot
		// method
		int bytecodeOffset;
		if (bootSrcMethod) {
			firstMethod = "<boot>";
			bytecodeOffset = 0;
		} else {
			bytecodeOffset = off;
		}

		// Skip this edge if destination node is a boot method
		if (secondMethod.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V")) {
			return null;
		}
		return firstMethod + "," + bytecodeOffset + "," + secondMethod;
	}

	// Computes a signature string for the edge.
	// This is supposed to match the format used in the pruned-callgraph file.
	public static String getEdgeSignature(CGNode srcMethod, CallSiteReference callsite, CGNode destNode) {
		// Get the first method's string
		String firstMethod = getMethodString(srcMethod);
		if (firstMethod == null) {
			return "";
		}
		
		// Get the second method's string
		String secondMethod = getMethodString(destNode);
		if (secondMethod == null) {
			return "";
		}
		
		// Record if this is a fakeRoot/boot method or not
		boolean bootSrcMethod = (firstMethod.equals("com/ibm/wala/FakeRootClass.fakeRootMethod:()V")
						|| firstMethod.equals("com/ibm/wala/FakeRootClass.fakeWorldClinit:()V"));
		
		// Finally put it all together
		String formattedOutputLine = formatFinalOutput(firstMethod, secondMethod, bootSrcMethod,
				callsite.getProgramCounter());
		if (formattedOutputLine == null) {
			return "";
		}
		return formattedOutputLine;
	}

	public static String getMethodString(CGNode srcMethod) {
		IMethod m1 = srcMethod.getMethod();
		TypeName t1 = m1.getDeclaringClass().getName();
		Selector sel1 = m1.getSelector();
		String name1 = sel1.getName().toString();
		String methodString = formatMethod(t1, name1, sel1);
		return methodString;
	}
}
