package org.example;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

public class VariableClassVisitor extends ClassVisitor {

    private String className;

    public VariableClassVisitor(ClassVisitor cv, String className){
        super(Opcodes.ASM5, cv);
        this.className = className;

    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String desc, String signature, String[] exceptions){
        MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
        return new VariableMethodVisitor(mv, name, className, desc);
    }



}
