package org.example;

import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import org.objectweb.asm.*;


public class IfStatementMethodVisitor extends MethodVisitor {

    private String methodName;
    private String className;
    private String desc;
    private int ifStatementCounter = 0;

    public IfStatementMethodVisitor(MethodVisitor mv, String name, String className, String desc){
        super(Opcodes.ASM5, mv);
        this.methodName = name;
        this.className = className;
        this.desc = desc;
    }

    @Override
    public void visitJumpInsn(int opcode, Label label) {
        if (opcode == Opcodes.IFEQ || opcode == Opcodes.IFNE || opcode == Opcodes.IFLT || opcode == Opcodes.IFGE ||
                opcode == Opcodes.IFGT || opcode == Opcodes.IFLE || opcode == Opcodes.IF_ICMPEQ || opcode == Opcodes.IF_ICMPNE ||
                opcode == Opcodes.IF_ICMPLT || opcode == Opcodes.IF_ICMPGE || opcode == Opcodes.IF_ICMPGT || opcode == Opcodes.IF_ICMPLE ||
                opcode == Opcodes.IF_ACMPEQ || opcode == Opcodes.IF_ACMPNE || opcode == Opcodes.IFNULL || opcode == Opcodes.IFNONNULL) {

            mv.visitFieldInsn(Opcodes.GETSTATIC, "java/lang/System", "out", "Ljava/io/PrintStream;");
            String ifStatementId = className + "." + methodName + desc + ":IF#" + ifStatementCounter;
            ifStatementCounter++;
            mv.visitLdcInsn("If statement: " + ifStatementId);
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/io/PrintStream", "println", "(Ljava/lang/String;)V", false);
        }
        super.visitJumpInsn(opcode, label);
    }

}