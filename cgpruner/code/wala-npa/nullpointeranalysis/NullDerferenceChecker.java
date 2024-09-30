package nullpointeranalysis;

import java.util.ArrayList;

import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ssa.*;

public class NullDerferenceChecker extends SSAInstruction.Visitor{
	private IMethod currentMethod;
    public static ArrayList<String> nullWarnings = new ArrayList<String>();
    public NullDerferenceChecker(CGNode n) {
		currentMethod = n.getMethod();
	}

	@Override
    public void visitGet(SSAGetInstruction ins) {
		Val v = Utils.getLElement(currentMethod, ins.getRef());
		if (v.baseRepresentation == ValBase.MAYBENULL) {
			nullWarnings.add("NPE_GET," 
					+ currentMethod.getReference().getSignature() + "," 
					+ ins.getDeclaredField().getSignature());
		}
    }  
    
    @Override
    public void visitInvoke(SSAInvokeInstruction ins) {
    	if (!ins.isStatic() && !ins.isSpecial()) {
    		Val v = Utils.getLElement(currentMethod, ins.getReceiver());
    		if (v.baseRepresentation == ValBase.MAYBENULL) {
    			nullWarnings.add("NPE_INV," 
    					+ currentMethod.getReference().getSignature() + "," 
    					+ ins.getDeclaredTarget().getSignature());
    		}
    	}
    }
    
    @Override
    public void visitPut(SSAPutInstruction ins) {
    	Val v = Utils.getLElement(currentMethod, ins.getRef());
		if (v.baseRepresentation == ValBase.MAYBENULL) {
			nullWarnings.add("NPE_PUT," 
					+ currentMethod.getReference().getSignature() + "," 
					+ ins.getDeclaredField().getSignature());
		}
    }    

}
