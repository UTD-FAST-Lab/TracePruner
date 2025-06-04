package org.example

import java.io.File
import java.io.PrintWriter
import java.io.Writer
import java.net.URL
import java.nio.file.Files

import scala.collection.compat.immutable
import scala.collection.mutable
import scala.io.Source
import scala.sys.process.Process

import org.apache.commons.io.FileUtils
import play.api.libs.json.Json
import play.api.libs.json.JsValue

import org.opalj.br.ClassFile
import org.opalj.br.FieldType
import org.opalj.br.FieldTypes
import org.opalj.br.MethodDescriptor
import org.opalj.br.ObjectType
import org.opalj.br.ReferenceType
import org.opalj.br.ReturnType
import org.opalj.br.analyses.Project
import org.opalj.br.analyses.SomeProject
import org.opalj.br.instructions.Instruction
import org.opalj.br.instructions.INVOKEDYNAMIC
import org.opalj.br.instructions.MethodInvocationInstruction

object DoopPCExtractor {

  private def resolveBridgeMethod(
        classFile: ClassFile,
        p: SomeProject,
        bridgeMethod: org.opalj.br.Method
    ): org.opalj.br.Method = {
        val methods = classFile.findMethod(bridgeMethod.name).filter { m ⇒
            !m.isBridge && (m.returnType match {
                case rt: ReferenceType ⇒ p.classHierarchy.isSubtypeOf(
                    rt, bridgeMethod.returnType.asReferenceType
                )
                case rt ⇒ rt == bridgeMethod.returnType
            })
        }
        assert(methods.size == 1)
        methods.head
    }


  private def computeCallSite(
        classFile: ClassFile,
        p: SomeProject,
        declaredTgt: String,
        number: Int,
        callerMethod: String,
        method: org.opalj.br.Method
    ): Unit = {
        val split = declaredTgt.split("""\.""")
        val declaredType = s"L${split.slice(0, split.size - 1).mkString("/")};"
        val name = split.last.replace("'", "")
        val declObjType = FieldType(declaredType)

        val getInstr: PartialFunction[Instruction, Instruction] = {
            // todo what about lambdas?
            case instr: MethodInvocationInstruction if (
                instr.name == name &&
                    (instr.declaringClass == declObjType ||
                        declObjType == ObjectType.Object && instr.declaringClass.isArrayType)
                ) ⇒ instr //&& instr.declaringClass == FieldType(declaredType) ⇒ instr // && instr.methodDescriptor == tgtMD ⇒ instr
            case instr: INVOKEDYNAMIC ⇒ instr
                //throw new Error()
        }


        val calls = method.body.get.collect(getInstr)


        if (calls.size <= number && method.isBridge) {
            computeCallSite(classFile, p, declaredTgt, number, callerMethod, resolveBridgeMethod(classFile, p, method))
        } else {
            assert(calls.size > number)
            val pc = calls(number).pc
            val lineNumber = method.body.get.lineNumber(pc)

            // write the pc and line number, number and instruction to a file
            val outputFile = new File(s"${callerMethod}_pc_${number}.txt")
            val writer = new PrintWriter(outputFile)
            writer.println(s"$pc")
            writer.close()

            // println(s"PC: $pc, Line Number: $lineNumber")
        }
      }

  def main(args: Array[String]): Unit = {

    val inputJar = new File(args(0))
    val callerMethod = args(1)
    val callerClass = args(2)
    val callerSignature = args(3)
    val jdkPath = new File(args(4))

    val declaredTgt = args(5)
    val number = args(6).toInt
    

    val project = Project(inputJar, jdkPath)

    project.classFile(ObjectType(callerClass)) match {
      case Some(cf) =>

        val md = MethodDescriptor(callerSignature)

        cf.findMethod(callerMethod, md) match {
          case Some(method) =>

            computeCallSite(cf, project, declaredTgt, number, callerMethod, method)

            // val split = declaredTgt.split("""\.""")
            // val declaredType = s"L${split.slice(0, split.size - 1).mkString("/")};"
            // val name = split.last.replace("'", "")
            // val declObjType = FieldType(declaredType)

            // val getInstr: PartialFunction[Instruction, Instruction] = {
            //     // todo what about lambdas?
            //     case instr: MethodInvocationInstruction if (
            //         instr.name == name &&
            //             (instr.declaringClass == declObjType ||
            //                 declObjType == ObjectType.Object && instr.declaringClass.isArrayType)
            //         ) ⇒ instr //&& instr.declaringClass == FieldType(declaredType) ⇒ instr // && instr.methodDescriptor == tgtMD ⇒ instr
            //     case instr: INVOKEDYNAMIC ⇒ instr
            //         //throw new Error()
            // }


            // val calls = method.body.get.collect(getInstr)


            // if (calls.size <= number && method.isBridge) {
            //     computeCallSite(declaredTgt, number, tgts, callerMethod, resolveBridgeMethod(callerOpal))
            // } else {
            //     assert(calls.size > number)
            //     val pc = calls(number).pc
            //     val lineNumber = callerOpal.body.get.lineNumber(pc)
            // }


            // assert(calls.size > number)
            // val pc = calls(number).pc
            // val lineNumber = method.body.get.lineNumber(pc)

            // println(s"PC: $pc, Line Number: $lineNumber")


          case None =>
            println(s"not found")
            return
        }







        // print the name of the methods that are in the class with out desired method name
        // for (method <- cf.findMethod(callerMethod)) {
            // get all of the instrcutions of the method and print the instruction with the pc
            
            // val getInstr: PartialFunction[Instruction, Instruction] = {
            //   // todo what about lambdas?
            //   case instr: MethodInvocationInstruction => instr
        
            //   case instr: INVOKEDYNAMIC ⇒ instr
            //       //throw new Error()
            // }

            // val calls = method.body.get.collect(getInstr)
            
            // for (call <- calls) {
            //   println(s"PC: ${call.pc}, Instruction: $call")
            // }
        // }
    }
  }
}