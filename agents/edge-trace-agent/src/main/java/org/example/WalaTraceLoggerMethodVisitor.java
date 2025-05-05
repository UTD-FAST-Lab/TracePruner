package org.example;

import org.objectweb.asm.*;

import java.util.HashMap;
import java.util.Map;




public class WalaTraceLoggerMethodVisitor extends MethodVisitor {

    private final WalaTraceLoggerAgent.AgentLevel agentLevel;
    private final WalaTraceLoggerAgent.LogLevel logLevel;
    private final String programDirPath;

    private String methodName;
    private String className;
    private String desc;
    private int ifStatementCounter = 0;
    

    public WalaTraceLoggerMethodVisitor(MethodVisitor mv, String name, String className, String desc, WalaTraceLoggerAgent.AgentLevel agentLevel, WalaTraceLoggerAgent.LogLevel logLevel, String programDirPath) {
        super(Opcodes.ASM5, mv);
        this.agentLevel = agentLevel;
        this.logLevel = logLevel;

        this.methodName = name;
        this.className = className;
        this.desc = desc;
        this.programDirPath = programDirPath;
    }


    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String descriptor, boolean isInterface) {
        

        if (name.equals("visitInvokeInternal")) {
            mv.visitVarInsn(Opcodes.ALOAD, 1);  // Load `instruction` 
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/InvocationLogger", "addInstruction",
                "(Ljava/lang/Object;)V",false);


        } else if (name.equals("processResolvedCall")){
            mv.visitVarInsn(Opcodes.ALOAD, 2);  // Load `instruction` 
            mv.visitVarInsn(Opcodes.ALOAD, 1);  // Load `src node` 
            mv.visitVarInsn(Opcodes.ALOAD, 3);  // Load 'Target node'
            mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/InvocationLogger", "writeTrace",
                "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;)V", false);
        }


        if (agentLevel == WalaTraceLoggerAgent.AgentLevel.FULL || agentLevel == WalaTraceLoggerAgent.AgentLevel.CG){

            if (owner.contains("wala")){


                // Load the individual parameters onto the stack
                mv.visitLdcInsn(className); // className
                mv.visitLdcInsn(methodName); // methodName
                mv.visitLdcInsn(desc);       // method descriptor
                mv.visitLdcInsn(owner);      // owner class
                mv.visitLdcInsn(name);       // method name being called
                mv.visitLdcInsn(descriptor); // method descriptor being called


                // Call the static logging method with the parameters
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/InvocationLogger", "addLineToTrace",
                                "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V",
                                false);
            }
        }

        super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }


    @Override
    public void visitJumpInsn(int opcode, Label label) {

        if (agentLevel == WalaTraceLoggerAgent.AgentLevel.FULL || agentLevel == WalaTraceLoggerAgent.AgentLevel.BRANCH) {

            if (className.contains("wala")) {

                if (opcode == Opcodes.IFEQ || opcode == Opcodes.IFNE || opcode == Opcodes.IFLT || opcode == Opcodes.IFGE ||
                    opcode == Opcodes.IFGT || opcode == Opcodes.IFLE || opcode == Opcodes.IF_ICMPEQ || opcode == Opcodes.IF_ICMPNE ||
                    opcode == Opcodes.IF_ICMPLT || opcode == Opcodes.IF_ICMPGE || opcode == Opcodes.IF_ICMPGT || opcode == Opcodes.IF_ICMPLE ||
                    opcode == Opcodes.IF_ACMPEQ || opcode == Opcodes.IF_ACMPNE || opcode == Opcodes.IFNULL || opcode == Opcodes.IFNONNULL) {
                    
                    
                    // Clone of the original jump to a merge point
                    Label originalJumpTarget = new Label();
                    mv.visitJumpInsn(opcode, originalJumpTarget);
                    
                    // "Else" branch logging
                    mv.visitLdcInsn(className);
                    mv.visitLdcInsn(methodName);
                    mv.visitLdcInsn(desc);
                    mv.visitLdcInsn(ifStatementCounter);
                    mv.visitInsn(Opcodes.ICONST_0);  // Push false (0) for "else" branch
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/InvocationLogger", "addLineToTrace",
                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IZ)V", false);
                    
                    // Jump to the end of instrumentation
                    Label endInstrumentationLabel = new Label();
                    mv.visitJumpInsn(Opcodes.GOTO, endInstrumentationLabel);
                    
                    // "If" branch logging
                    mv.visitLabel(originalJumpTarget);
                    mv.visitLdcInsn(className);
                    mv.visitLdcInsn(methodName);
                    mv.visitLdcInsn(desc);
                    mv.visitLdcInsn(ifStatementCounter);
                    mv.visitInsn(Opcodes.ICONST_1);  // Push true (1) for "if" branch
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "org/example/InvocationLogger", "addLineToTrace",
                            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IZ)V", false);
                    
                    // Jump to the original target
                    mv.visitJumpInsn(Opcodes.GOTO, label);
                    
                    // Original jump instruction - this is needed when recording is disabled
                    mv.visitJumpInsn(opcode, label);
                    
                    // End of instrumentation
                    mv.visitLabel(endInstrumentationLabel);
                    
                    ifStatementCounter++;
                    return;
                }
            }
        }
 
        super.visitJumpInsn(opcode, label);
    }

}