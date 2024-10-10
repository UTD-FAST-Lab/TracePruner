package org.example;

import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public class PrintMessageMethodVisitor extends MethodVisitor {

    private String methodName;
    private String className;
    private String desc;

    public PrintMessageMethodVisitor(MethodVisitor mv,String name, String className, String desc){
        super(Opcodes.ASM5, mv);
        this.methodName = name;
        this.className = className;
        this.desc = desc;
    }

    @Override
    public void visitMethodInsn(int opcode, String owner, String name, String descriptor, boolean isInterface) {
        // Print call graph information

        if (owner.startsWith("jayhorn")){
            mv.visitFieldInsn(Opcodes.GETSTATIC, "java/lang/System", "out", "Ljava/io/PrintStream;");
            mv.visitLdcInsn("AgentLogger|CG_edge: " + className.replace('/', '.') + "." + methodName + " " + desc + " -> " + owner.replace('/', '.') + "." + name + " " + descriptor);
            mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, "java/io/PrintStream", "println", "(Ljava/lang/String;)V", false);
        }

        super.visitMethodInsn(opcode, owner, name, descriptor, isInterface);
    }

}