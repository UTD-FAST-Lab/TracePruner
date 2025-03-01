package org.example;

import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import org.objectweb.asm.*;

import java.util.HashMap;
import java.util.Map;




public class IntegratedMethodVisitor extends MethodVisitor {

    private final IntegratedLoggerAgent.AgentLevel agentLevel;
    private final IntegratedLoggerAgent.LogLevel logLevel;

    private String methodName;
    private String className;
    private String desc;
    private int ifStatementCounter = 0;
    private final Map<Integer, String> localVarNames = new HashMap<>();


    // call garph range

    // jcg driver

    private static final String START_SOURCE_CLASS = "WalaJCGAdapter$";
    private static final String START_SOURCE_METHOD = "serializeCG";
    private static final String START_SOURCE_DESC = "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)J";
    private static final String START_TARGET_CLASS = "com/ibm/wala/ipa/callgraph/impl/Util";
    private static final String START_TARGET_METHOD = "makeZeroCFABuilder";
    private static final String START_TARGET_DESC = "(Lcom/ibm/wala/classLoader/Language;Lcom/ibm/wala/ipa/callgraph/AnalysisOptions;Lcom/ibm/wala/ipa/callgraph/IAnalysisCacheView;Lcom/ibm/wala/ipa/cha/IClassHierarchy;Lcom/ibm/wala/ipa/callgraph/AnalysisScope;)Lcom/ibm/wala/ipa/callgraph/propagation/SSAPropagationCallGraphBuilder;";

    private static final String END_SOURCE_CLASS = "WalaJCGAdapter$";
    private static final String END_SOURCE_METHOD = "serializeCG";
    private static final String END_SOURCE_DESC = "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)J";
    private static final String END_TARGET_CLASS = "com/ibm/wala/ipa/cha/ClassHierarchy";
    private static final String END_TARGET_METHOD = "resolveMethod";
    private static final String END_TARGET_DESC = "(Lcom/ibm/wala/types/MethodReference;)Lcom/ibm/wala/classLoader/IMethod;";

    // cgPruner driver

    // private static final String START_SOURCE_CLASS = "com/example/WalaCallgraph";
    // private static final String START_SOURCE_METHOD = "main";
    // private static final String START_SOURCE_DESC = "([Ljava/lang/String;)V";
    // private static final String START_TARGET_CLASS = "com/ibm/wala/ipa/callgraph/impl/Util";
    // private static final String START_TARGET_METHOD = "makeZeroCFABuilder";
    // private static final String START_TARGET_DESC = "(Lcom/ibm/wala/classLoader/Language;Lcom/ibm/wala/ipa/callgraph/AnalysisOptions;Lcom/ibm/wala/ipa/callgraph/IAnalysisCacheView;Lcom/ibm/wala/ipa/cha/IClassHierarchy;Lcom/ibm/wala/ipa/callgraph/AnalysisScope;)Lcom/ibm/wala/ipa/callgraph/propagation/SSAPropagationCallGraphBuilder;";

    // private static final String END_SOURCE_CLASS = "WalaJCGAdapter$";
    // private static final String END_SOURCE_METHOD = "serializeCG";
    // private static final String END_SOURCE_DESC = "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;ZLjava/lang/String;)J";
    // private static final String END_TARGET_CLASS = "com/ibm/wala/ipa/cha/ClassHierarchy";
    // private static final String END_TARGET_METHOD = "resolveMethod";
    // private static final String END_TARGET_DESC = "(Lcom/ibm/wala/types/MethodReference;)Lcom/ibm/wala/classLoader/IMethod;";


    // Flag to control recording
    // private static boolean recording = false;




    public IntegratedMethodVisitor(MethodVisitor mv, String name, String className, String desc, IntegratedLoggerAgent.AgentLevel agentLevel, IntegratedLoggerAgent.LogLevel logLevel){
        super(Opcodes.ASM5, mv);
        this.agentLevel = agentLevel;
        this.logLevel = logLevel;

        this.methodName = name;
        this.className = className;
        this.desc = desc;
    }


    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String descriptor, boolean isInterface) {

        // start recording from a certain point to a certian point
        if (className.equals(START_SOURCE_CLASS) && methodName.equals(START_SOURCE_METHOD) && desc.equals(START_SOURCE_DESC)
            && owner.equals(START_TARGET_CLASS) && name.equals(START_TARGET_METHOD) && descriptor.equals(START_TARGET_DESC)) {
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/RecordingControl", "startRecording", "()V", false);
        }
        if (className.equals(END_SOURCE_CLASS) && methodName.equals(END_SOURCE_METHOD) && desc.equals(END_SOURCE_DESC)
            && owner.equals(END_TARGET_CLASS) && name.equals(END_TARGET_METHOD) && descriptor.equals(END_TARGET_DESC)) {
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/RecordingControl", "stopRecording", "()V", false);
        }


        


        // Print the start and end of an edge trace
        if (name.equals("visitInvokeInternal")) {
            mv.visitLdcInsn(name);
            mv.visitVarInsn(Opcodes.ALOAD, 1);  // Load `instruction` (assumed to be the first argument)
            mv.visitInsn(Opcodes.ACONST_NULL);  // Push `null`
            mv.visitInsn(Opcodes.ACONST_NULL);  // Push `null`
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logBoundaries",
                "(Ljava/lang/String;Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;)V",false);
        } else if (name.equals("processResolvedCall")){
            mv.visitLdcInsn(name);
            mv.visitVarInsn(Opcodes.ALOAD, 2);  // Load `instruction` 
            mv.visitVarInsn(Opcodes.ALOAD, 1);  // Load `src node` 
            mv.visitVarInsn(Opcodes.ALOAD, 3);  // Load 'Target node'
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logBoundaries",
                "(Ljava/lang/String;Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;)V",false);
        }

        


    
        if (agentLevel == IntegratedLoggerAgent.AgentLevel.FULL || agentLevel == IntegratedLoggerAgent.AgentLevel.CG){

            if (owner.contains("wala")){

                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/RecordingControl", "isRecording", "()Z", false);

                Label skipLabel = new Label();
                mv.visitJumpInsn(Opcodes.IFEQ, skipLabel); // Skip logging if not recording
                // mv.visitInsn(Opcodes.POP);

                // Load the individual parameters onto the stack
                mv.visitLdcInsn(className); // className
                mv.visitLdcInsn(methodName); // methodName
                mv.visitLdcInsn(desc);       // method descriptor
                mv.visitLdcInsn(owner);      // owner class
                mv.visitLdcInsn(name);       // method name being called
                mv.visitLdcInsn(descriptor); // method descriptor being called


                // Call the static logging method with the parameters
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logCGEdge",
                                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V",
                                false);

                mv.visitLabel(skipLabel);
            }
        }

        

        super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }


    @Override
    public void visitJumpInsn(int opcode, Label label) {

        
        if (agentLevel == IntegratedLoggerAgent.AgentLevel.FULL || agentLevel == IntegratedLoggerAgent.AgentLevel.BRANCH){

            if (className.contains("wala")){

                if (opcode == Opcodes.IFEQ || opcode == Opcodes.IFNE || opcode == Opcodes.IFLT || opcode == Opcodes.IFGE ||
                        opcode == Opcodes.IFGT || opcode == Opcodes.IFLE || opcode == Opcodes.IF_ICMPEQ || opcode == Opcodes.IF_ICMPNE ||
                        opcode == Opcodes.IF_ICMPLT || opcode == Opcodes.IF_ICMPGE || opcode == Opcodes.IF_ICMPGT || opcode == Opcodes.IF_ICMPLE ||
                        opcode == Opcodes.IF_ACMPEQ || opcode == Opcodes.IF_ACMPNE || opcode == Opcodes.IFNULL || opcode == Opcodes.IFNONNULL) {

                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/RecordingControl", "isRecording", "()Z", false);

                    Label skipLabel = new Label();
                    mv.visitJumpInsn(Opcodes.IFEQ, skipLabel); // Skip logging if not recording

                    mv.visitLdcInsn(className);      // Push className onto the stack
                    mv.visitLdcInsn(methodName);     // Push methodName onto the stack
                    mv.visitLdcInsn(desc);           // Push desc onto the stack
                    mv.visitLdcInsn(ifStatementCounter); // Push ifStatementCounter onto the stack

                    // Increment the ifStatementCounter for the next statement
                    ifStatementCounter++;

                    // Call the custom logging method with the new parameters
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logBranch",
                                    "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V",
                                    false);

                    mv.visitLabel(skipLabel);

                }
            }
        }
        
        super.visitJumpInsn(opcode, label);
    }


     // Capture local variable names (only available if compiled with debug info)
    @Override
    public void visitLocalVariable(String name, String descriptor, String signature, Label start, Label end, int index) {
        localVarNames.put(index, name); // Map variable index (slot) to variable name
        super.visitLocalVariable(name, descriptor, signature, start, end, index);
    }

    // // Intercept STORE instructions (local variable writes)
    @Override
    public void visitVarInsn(int opcode, int varIndex) {
        super.visitVarInsn(opcode, varIndex);


        if (agentLevel == IntegratedLoggerAgent.AgentLevel.FULL || agentLevel == IntegratedLoggerAgent.AgentLevel.VAR){

            if (className.contains("wala")){

                if (opcode >= Opcodes.ISTORE && opcode <= Opcodes.ASTORE) {

                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/RecordingControl", "isRecording", "()Z", false);
                    Label skipLabel = new Label();
                    mv.visitJumpInsn(Opcodes.IFEQ, skipLabel); // Skip logging if not recording

                    // Get the variable name for logging
                    String varName = localVarNames.getOrDefault(varIndex, ":VAR#" + varIndex);

                    // Push parameters onto the stack in order for the method call
                    mv.visitLdcInsn(className);  // Push className
                    mv.visitLdcInsn(methodName); // Push methodName
                    mv.visitLdcInsn(desc);       // Push desc
                    mv.visitLdcInsn(varName);    // Push varName



                    // Load the variable value onto the stack and call the corresponding append method
                    switch (opcode) {
                        case Opcodes.ISTORE:
                            mv.visitVarInsn(Opcodes.ILOAD, varIndex);
                            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logVariable",
                                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V", false);
                            break;
                        case Opcodes.FSTORE:
                            mv.visitVarInsn(Opcodes.FLOAD, varIndex);
                            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logVariable",
                                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;F)V", false);
                            break;
                        case Opcodes.DSTORE:
                            mv.visitVarInsn(Opcodes.DLOAD, varIndex);
                            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logVariable",
                                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;D)V", false);
                            break;
                        case Opcodes.LSTORE:
                            mv.visitVarInsn(Opcodes.LLOAD, varIndex);
                            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logVariable",
                                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;J)V", false);
                            break;
                        case Opcodes.ASTORE:
                            mv.visitVarInsn(Opcodes.ALOAD, varIndex);
                            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/AgentLogger", "logVariable",
                                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/Object;)V", false);
                            break;
                    }


                    mv.visitLabel(skipLabel);
                }
            }
        }

    }

}