package org.example;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

// import java.io.BufferedWriter;
// import java.io.FileWriter;
// import java.io.IOException;

public class LogMethodClassVisitor extends ClassVisitor {

    private String className;

    public LogMethodClassVisitor(ClassVisitor cv, String className){
        super(Opcodes.ASM5, cv);
        this.className = className;

    }

    @Override
    public MethodVisitor visitMethod(int access, String name, String desc, String signature, String[] exceptions){
        MethodVisitor mv = super.visitMethod(access, name, desc, signature, exceptions);
        return new PrintMessageMethodVisitor(mv, name, className, desc);
    }



}
