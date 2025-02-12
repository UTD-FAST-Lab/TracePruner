package org.example;

import org.objectweb.asm.*;

import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.security.ProtectionDomain;

public class MethodExecutionLoggerTransformer implements ClassFileTransformer  {


        public MethodExecutionLoggerTransformer(){

        }

        @Override
        public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                                ProtectionDomain protectionDomain, byte[] classfileBuffer) throws IllegalClassFormatException {

            if (className != null && className.contains("wala")) {
                System.out.println("Transforming class: " + className);
                ClassReader cr = new ClassReader(classfileBuffer);
                ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_FRAMES);
                ClassVisitor cv = new LogMethodClassVisitor(cw, className);
                cr.accept(cv, 0);
                return cw.toByteArray();
            }
            return null;
        }
}
