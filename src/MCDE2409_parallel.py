#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(3,1)-MCDE code.




@author: Stefano Paggi
"""
import numpy as np



import scipy.linalg
import numpy as np
from collections import Counter
import time
from collections import defaultdict
from operator import itemgetter
from scipy.sparse import lil_matrix, csr_matrix
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# np.random.seed(42)

#%% the class

class MCDE():
    """
        Attributes:
            zero_tol : float
                Threshold below which matrix elements in the effective Hamiltonian
                are treated as zero.
            
            zero_tol_evals : float
                Threshold below which eigenvalues are treated as zero.
            
            lanczos_tol : float
                Convergence tolerance for the Lanczos algorithm.
            
            remove_single_values : bool
                If ``True``, isolated diagonal elements are removed from the
                effective Hamiltonian.
            
            lanczos : bool
                If ``True``, use the Lanczos algorithm for diagonalization.
            
            exactDiag : bool
                If ``True``, perform exact diagonalization.
            
            spinOptimized : bool
                If ``True``, perform a spin-optimized calculation.
            
            fullCalculation : bool
                If ``True``, construct the full double-basis Hamiltonian.
            
            secondBorn : bool
                If ``True``, perform a Second-Born calculation.
            
            mcde : bool
                If ``True``, perform an MCDE calculation.
            
            do_sparse : bool
                If ``True``, force the use of sparse matrix methods.
            
            do_auto_sparse : bool
                If ``True``, automatically switch to the sparse implementation
                when the basis size exceeds ``do_auto_sparse_basis_threshold``.
            
            do_auto_sparse_basis_threshold : int
                Basis-size threshold above which the sparse implementation is
                automatically selected when ``do_auto_sparse`` is enabled.
            
            data_type_sparse : numpy.dtype
                Numerical precision used for sparse matrix calculations.
            
            reduce_evecs_to_1body : bool
                If ``True``, discard the three-particle component of the
                eigenvectors after diagonalization, retaining only the
                one-particle contribution.
            
            sparse_tol : float
                Threshold below which matrix elements are omitted during sparse
                effective Hamiltonian construction.
            
            shift_virtual_energy : float
                Energy shift applied to virtual orbitals in the three-particle
                block of the effective Hamiltonian.
            
            kernel_run : bool
                Indicates whether the kernel calculation has been executed.

    """
    # object saving the three and one body part of matrix
    class Nspace():
        
        def __init__(self,d1,d3):
            self.d1=d1
            self.d3=d3
            
        def v(self,n):
            if n>=len(self.d1):
                return self.d3[n-len(self.d1)]
            return self.d1[n]
        
        def stats(self):
            print("1-particle length: "+str(len(self.d1)))
            print("3-particle length: "+str(len(self.d3)))
            print("total length: "+str(len(self.d1)+len(self.d3)))
            
        def translateToAOLabels(self,n,ao_labels):
            vec=self.v(n)
            if isinstance(vec,np.ndarray):
                a=ao_labels[vec[0]]
                b=ao_labels[vec[1]]
                c=ao_labels[vec[2]]
                
                return "{"+a+", "+b+", "+c+"}"
            return ao_labels[vec]
        
        def returnNumLabels(self,n):
            vec=self.v(n)
            if isinstance(vec,np.ndarray):
                a=str(vec[0])
                b=str(vec[1])
                c=str(vec[2])
                
                return "{"+a+", "+b+", "+c+"}"
            return str(vec)
                
                
            
    
    class AuxillaryFunctions():
        
        #def __init__(self):
        
        @staticmethod
        def eig(matrix):
            return np.linalg.eigh(matrix)
        
        @staticmethod
        def remove_isolated_diagonals(A,spaceObj,remove_single_values):
            if not remove_single_values:
                return A,spaceObj
            A = np.array(A)
            keep = []
            for i in range(A.shape[0]):
                if A[i, i] == 0:
                    keep.append(i)
                else:
                    row = np.copy(A[i, :])
                    col = np.copy(A[:, i])
                    row[i] = 0
                    col[i] = 0
                    if np.any(row) or np.any(col):
                        keep.append(i)
            # Keep only rows and columns that are not isolated diagonals
            A_new = A[np.ix_(keep, keep)]
            
            
            d1=[]
            d3=[]
            #spaceObj.stats()
            for x in keep:
                if x>=len(spaceObj.d1):
                    d3.append(spaceObj.v(x))
                else:
                    d1.append(spaceObj.v(x))
            return A_new, MCDE.Nspace(np.array(d1), np.array(d3))
        
        @staticmethod
        def remove_isolated_diagonals_sparse(A,spaceObj,remove_single_values):
            if not remove_single_values:
                return A,spaceObj
            # Step 1: Count nonzeros per row and per column
            row_nnz = np.diff(A.indptr)           # CSR: number of nonzeros per row
            col_nnz = np.diff(A.tocsc().indptr)  # CSC: number of nonzeros per column
            
            # Step 2: Identify diagonal indices
            diag_idx = np.arange(A.shape[0])
            
            # Step 3: Mask for diagonals that are isolated
            isolated_diag_mask = (row_nnz == 1) & (col_nnz == 1)
            
            # Step 4: Zero out isolated diagonals
            A[diag_idx[isolated_diag_mask], diag_idx[isolated_diag_mask]] = 0
            
            # Step 5: Remove stored zeros
            A.eliminate_zeros()
            
            # print("bing")
            d1=[]
            d3=[]
            #spaceObj.stats()
            for x,v in enumerate(isolated_diag_mask):
                if v:
                    continue
                if x>=len(spaceObj.d1):
                    d3.append(spaceObj.v(x))
                else:
                    d1.append(spaceObj.v(x))
                    
            # print(len(isolated_diag_mask))
            return A,MCDE.Nspace(np.array(d1), np.array(d3))
            
    def __init__(self,nBas,nO,moEn,erimo,verbose=0):
       
        
        self.starttime=time.perf_counter()
        self.intermediatetime=self.starttime
        
        self.verbose=verbose
        
        self.verbose1("_________________________")
        self.verbose1(" _      ____  ____  _____")
        self.verbose1("/ \\__/|/   _\\/  _ \\/  __/")
        self.verbose1("| |\\/|||  /  | | \\||  \\  ")
        self.verbose1("| |  |||  \\__| |_/||  /_ ")
        self.verbose1("\\_/  \\|\\____/\\____/\\____\\")
        self.verbose1("      v2409                 ")
        self.verbose1("_________________________")

        
        self.zero_tol=1e-10 #when values should be set to zero in the effective Hamiltonian
        self.zero_tol_evals=1e-16 #when values should be set to zero in the eigenvalues
        self.lanczos_tol=1e-12 #tolerance of Lanczos convergence
        self.remove_single_values=True #removes isolated diagonals from effective Hamiltonian
        self.lanczos=False # do lanczos algorithm
        self.exactDiag=True # do exact diagonalization
        self.spinOptimized=True # do a spinoptimized calculation
        self.fullCalculation=False # do the full double basis hamiltonian
        self.secondBorn=False # compute SB calculation
        self.mcde=True # compute MCDE calculation
        self.do_sparse=False
        self.do_auto_sparse=True # does sparse spinoptimized if the basis size is bigger than a threshold
        self.do_auto_sparse_basis_threshold=3000 # if the spin optimized basis is larger than 3000, the sparse method will be used if do_auto_sparse is set to true
        self.data_type_sparse=np.float64 # precision for the sparse matrix approach
        self.reduce_evecs_to_1body=False # discards the 3particle part of the eigenvectors, as they are not needed for the calculation of the spectrum
        self.sparse_tol=1e-8 #put values to zero in eff hamiltonian sparse algorithm when smaller than that
        
        self.shift_virtual_energy = 0 #energy shift in virtual energies in the body
        
        self.kernel_run=False # did you run the kernel?
        
        #self.mo_coeff = mocoeff
        self.mo_en = moEn
        self.eri_mo = erimo
        self.eri_mo_gabi =self.eri_mo.transpose((0,2,3,1))
        self.eri_mo_W = None
        self.eri_mo_gabi_W=None
        if self.eri_mo_W is not None:
            self.eri_mo_gabi_W=self.eri_mo_W.transpose((0,2,3,1))
        self.nO = nO
        self.nBas = nBas
        self.nV= nBas-nO
        
        estimate=int(2*int(self.nV*self.nO*(self.nO+1)/2+self.nO*self.nV*(self.nV+1)/2)-2*self.nO*self.nV+self.nBas)
        
        self.estimate_spin_opt_ham = estimate
        
        self.verbose2("Estimated size of spin Opt Hamiltonian basis: " + str(estimate))
        
        self.verbose4("nBas: "+str(nBas))
        self.verbose4("nO: "+str(self.nO))
        self.verbose4("nV: "+str(self.nV))
        
        self.iterations=max(int(estimate*.75),100) #lanczos iterations
        
        self.compare=[]
        self.compareshoulder=[]
        
    
    # cut corners everywhere
    def sparkurs(self):
        self.exactDiag=False
        self.data_type_sparse=np.float32
        self.do_sparse=True
    
    def kernel(self):
        
        self.verbose2("kernel start")
        self.verbose2(f"MCDE? {self.mcde}") # compute MCDE calculation
        self.verbose2(f"Second Born? {self.secondBorn}") # compute SB calculation
        self.verbose3(f"Spin Adapted Ham? {self.spinOptimized}") # do a spinoptimized calculation
        self.verbose3(f"Remove isolated diagonals from spin adapted Ham? {self.remove_single_values}") #removes isolated diagonals from effective Hamiltonian
        self.verbose3(f"Full eff Ham? {self.fullCalculation}") # do the full double basis hamiltonian
        self.verbose3(f"Effective Hamiltonian tolerance: {self.zero_tol}") #when values should be set to zero in the effective Hamiltonian
        
        if self.shift_virtual_energy != 0:
            self.verbose3(f"Shift of 3p virtual QP: {self.shift_virtual_energy}")
        
        if self.lanczos:
            self.verbose3(f"Lanczos algorithm? {self.lanczos}") # do lanczos algorithm
            self.verbose3(f"Lanczos convergence: {self.lanczos_tol}")#tolerance of Lanczos convergence
        
        if self.exactDiag:
            self.verbose3(f"Exact diagonalization? {self.exactDiag}") # do exact diagonalization
            self.verbose3(f"Eigenvalue tolerance: {self.zero_tol_evals}") #when values should be set to zero in the eigenvalues
            self.verbose3(f"Reduce eigenvectors to 1body part: {self.reduce_evecs_to_1body}") #only one body part needed for plotting
        
        if self.do_auto_sparse:
            if self.do_auto_sparse_basis_threshold<self.estimate_spin_opt_ham:
                self.do_sparse=True
                self.verbose3(f"Sparse method activated since basis size {self.estimate_spin_opt_ham} is larger than threshold {self.do_auto_sparse_basis_threshold}")
            else:
                self.verbose3(f"Sparse method: {self.do_sparse}")
        if self.do_sparse:
            self.verbose3(f"Sparse tolerance: {self.sparse_tol}")
            self.verbose3(f"sparse matrix datatype: {self.data_type_sparse}")
            
        if not self.do_sparse:    
            self.vout=self.vbar()
        self.nspace = self.g03spaceSelection()
        secondBorn=self.secondBorn
        calcMCDE=self.mcde
        full = self.fullCalculation
        exactdiag=self.exactDiag
        lanczos=self.lanczos
        spinOptimized=self.spinOptimized
        
        if full:
            self.sigma3s=self.sigma3shoulder()
            self.HeffSBmat=self.createSecondBornEffectiveHam()
            if calcMCDE:
                self.sigma3=self.g3sigma()
                # sets self.HeffMCDEmat
                self.HeffMCDE()
                if exactdiag:
                    self.HeffMCDEExactDiagEvals,self.HeffMCDEExactDiagEvecs=self.exactDiagonalization(self.HeffMCDEmat)
                if lanczos:
                    self.HeffMCDEAcoeff,self.HeffMCDEBcoeff=self.LanczosAlgorithm(self.HeffMCDEmat,self.nBas*2)
            if secondBorn:
                self.HeffSecondBorn()
                if exactdiag:
                    self.HeffSBxactDiagEvals,self.HeffSBxactDiagEvecs=self.exactDiagonalization(self.HeffSBmat)
                if lanczos:
                    self.HeffSBAcoeff,self.HeffSBBcoeff=self.LanczosAlgorithm(self.HeffSBmat,self.nBas*2)
        
        #spin free variant
        if spinOptimized:
            if not self.do_sparse:
                if secondBorn:
                    self.HeffMCDESpinAdaptmat,self.HeffSBSpinAdaptmat=self.spinAdaptedMCDE(secondBorn)
                    if exactdiag:
                        self.HeffMCDESpinAdaptExactDiagEvals,self.HeffMCDESpinAdaptExactDiagEvecs=self.exactDiagonalization(self.HeffMCDESpinAdaptmat)
                        self.HeffSBSpinAdaptExactDiagEvals,self.HeffSBSpinAdaptExactDiagEvecs=self.exactDiagonalization(self.HeffSBSpinAdaptmat)
                    if lanczos:
                        self.HeffMCDESpinAdaptAcoeff,self.HeffMCDESpinAdaptBcoeff=self.LanczosAlgorithm(self.HeffMCDESpinAdaptmat,self.nBas)
                        self.HeffSBSpinAdaptAcoeff,self.HeffSBSpinAdaptBcoeff=self.LanczosAlgorithm(self.HeffSBSpinAdaptmat,self.nBas)
                else:
                    self.HeffMCDESpinAdaptmat=self.spinAdaptedMCDE(secondBorn)
                    if exactdiag:
                        self.HeffMCDESpinAdaptExactDiagEvals,self.HeffMCDESpinAdaptExactDiagEvecs=self.exactDiagonalization(self.HeffMCDESpinAdaptmat)
                    if lanczos:
                        self.HeffMCDESpinAdaptAcoeff,self.HeffMCDESpinAdaptBcoeff=self.LanczosAlgorithm(self.HeffMCDESpinAdaptmat,self.nBas)
            else:
                if secondBorn and calcMCDE:
                    self.HeffMCDESpinAdaptSparsemat,self.HeffSBSpinAdaptSparsemat=self.spinAdaptedMCDESparse(secondBorn)
                    if exactdiag:
                        self.HeffMCDESpinAdaptSparseEvals,self.HeffMCDESpinAdaptSparseEvecs=self.exactDiagonalizationSparse(self.HeffMCDESpinAdaptSparsemat)
                        self.HeffSBSpinAdaptSparseEvals,self.HeffSBSpinAdaptSparseEvecs=self.exactDiagonalizationSparse(self.HeffSBSpinAdaptSparsemat)
                elif secondBorn and not calcMCDE:
                    self.HeffSBSpinAdaptSparsemat=self.spinAdaptedMCDESparse(secondBorn)
                    if exactdiag:                    
                        self.HeffSBSpinAdaptSparseEvals,self.HeffSBSpinAdaptSparseEvecs=self.exactDiagonalizationSparse(self.HeffSBSpinAdaptSparsemat)
                elif calcMCDE and not secondBorn:
                    self.HeffMCDESpinAdaptSparsemat=self.spinAdaptedMCDESparse(secondBorn)
                    if exactdiag:
                        self.HeffMCDESpinAdaptSparseEvals,self.HeffMCDESpinAdaptSparseEvecs=self.exactDiagonalizationSparse(self.HeffMCDESpinAdaptSparsemat)
        self.kernel_run=True
        self.timed("kernel end", 2)
        self.verbose1("_________________________")
        self.verbose1("_________________________")
            
        
        
    
        
        # self.heff_evals,self.heff_evecs,self.heff_sigma3Shoulder,self.heff_sigma3Body=self.Heff()
        
        # if secondBorn:
        #     self.heff,self.heffSB=self.spinTransformedMCDE(secondBorn=True)
        #     if self.lanczos:
        #         self.a_coeff_SB,self.b_coeff_SB=self.LanczosAlgorithm(self.heffSB)
        #     else:
        #         self.heff_evals_SB,self.heff_evecs_SB=self.AuxillaryFunctions.eig(self.heffSB)
        #         self.heff_evals_SB=np.where(np.abs(self.heff_evals_SB) < self.zero_tol_evals, 0.0, self.heff_evals_SB)
        #         self.sigma3s=self.sigma3shoulder()
        #         self.heffFull_evals_SB,self.heffFull_evecs_SB=self.HeffSB()
        
            
        
    
    # return eigenvectors and Hamiltonians based on parameters
    def result(self):
        if not self.kernel_run:
            raise ValueError("kernel() not run")
        secondBorn=self.secondBorn
        calcMCDE=self.mcde
        full = self.fullCalculation
        exactdiag=self.exactDiag
        lanczos=self.lanczos
        spinOptimized=self.spinOptimized
        
        if full:
            if calcMCDE and secondBorn:
                if exactdiag and lanczos:
                    return self.HeffMCDEExactDiagEvals,self.HeffMCDEExactDiagEvecs,self.HeffSBxactDiagEvals,self.HeffSBxactDiagEvecs,self.HeffMCDEAcoeff,self.HeffMCDEBcoeff,self.HeffSBAcoeff,self.HeffSBBcoeff,self.HeffMCDEmat,self.HeffSBmat
                if exactdiag:
                    return self.HeffMCDEExactDiagEvals,self.HeffMCDEExactDiagEvecs,self.HeffSBxactDiagEvals,self.HeffSBxactDiagEvecs,self.HeffMCDEmat,self.HeffSBmat
                if lanczos:
                    return self.HeffMCDEAcoeff,self.HeffMCDEBcoeff,self.HeffSBAcoeff,self.HeffSBBcoeff,self.HeffMCDEmat,self.HeffSBmat
                return self.HeffMCDEmat,self.HeffSBmat
            if calcMCDE:
                if exactdiag and lanczos:
                    return self.HeffMCDEExactDiagEvals,self.HeffMCDEExactDiagEvecs,self.HeffMCDEAcoeff,self.HeffMCDEBcoeff,self.HeffMCDEmat
                if exactdiag:
                    return self.HeffMCDEExactDiagEvals,self.HeffMCDEExactDiagEvecs,self.HeffMCDEmat
                if lanczos:
                    return self.HeffMCDEAcoeff,self.HeffMCDEBcoeff,self.HeffMCDEmat
                return self.HeffMCDEmat
            if secondBorn:
                if exactdiag and lanczos:
                    return self.HeffSBxactDiagEvals,self.HeffSBxactDiagEvecs,self.HeffSBAcoeff,self.HeffSBBcoeff,self.HeffSBmat
                if exactdiag:
                    return self.HeffSBxactDiagEvals,self.HeffSBxactDiagEvecs,self.HeffSBmat
                if lanczos:
                    return self.HeffSBAcoeff,self.HeffSBBcoeff,self.HeffSBmat
                return self.HeffSBmat
        
        #spin free variant
        if spinOptimized:
            if not self.do_sparse:
                if secondBorn:
                    if exactdiag:
                        return self.HeffMCDESpinAdaptExactDiagEvals,self.HeffMCDESpinAdaptExactDiagEvecs,self.HeffSBSpinAdaptExactDiagEvals,self.HeffSBSpinAdaptExactDiagEvecs,self.HeffMCDESpinAdaptmat,self.HeffSBSpinAdaptmat
                    if lanczos:
                        return self.HeffMCDESpinAdaptAcoeff,self.HeffMCDESpinAdaptBcoeff,self.HeffSBSpinAdaptAcoeff,self.HeffSBSpinAdaptBcoeff,self.HeffMCDESpinAdaptmat,self.HeffSBSpinAdaptmat
                    return self.HeffMCDESpinAdaptmat,self.HeffSBSpinAdaptmat
                if calcMCDE:
                    if exactdiag:
                        return self.HeffMCDESpinAdaptExactDiagEvals,self.HeffMCDESpinAdaptExactDiagEvecs,self.HeffMCDESpinAdaptmat
                    if lanczos:
                        return self.HeffMCDESpinAdaptAcoeff,self.HeffMCDESpinAdaptBcoeff,self.HeffMCDESpinAdaptmat
                    return self.HeffMCDESpinAdaptmat
            else:
                if secondBorn and calcMCDE:
                    if exactdiag:
                        return self.HeffMCDESpinAdaptSparseEvals,self.HeffMCDESpinAdaptSparseEvecs,self.HeffSBSpinAdaptSparseEvals,self.HeffSBSpinAdaptSparseEvecs,self.HeffMCDESpinAdaptSparsemat,self.HeffSBSpinAdaptSparsemat
                    return self.HeffMCDESpinAdaptSparsemat,self.HeffSBSpinAdaptSparsemat
                if calcMCDE and not secondBorn:
                    if exactdiag:
                        return self.HeffMCDESpinAdaptSparseEvals,self.HeffMCDESpinAdaptSparseEvecs,self.HeffMCDESpinAdaptSparsemat
                    return self.HeffMCDESpinAdaptSparsemat
                if secondBorn and not calcMCDE:
                    if exactdiag:
                        return self.HeffSBSpinAdaptSparseEvals,self.HeffSBSpinAdaptSparseEvecs,self.HeffSBSpinAdaptSparsemat
                    return self.HeffSBSpinAdaptSparsemat
                
    def saveMCDE(self,filename=None):
        
        if filename is None:
            filename=str(time.time())
        np.savez(filename+".npz",mcde=self)
        self.verbose2("MCDE object saved to " + filename+".npz")
    
    @staticmethod        
    def loadMCDE(filename):
        data = np.load(filename+".npz", allow_pickle=True)
        print("MCDE object loaded from " + filename+".npz")
        return data['mcde'].item()                

    
    def c2g(self,ind):
        return [ind[0],ind[2],ind[3],ind[1]]        
    def g2c(self,ind):
        return [ind[0],ind[3],ind[1],ind[2]]     
    
    def verbose1(self,txt):
        if self.verbose >= 1:
            print(txt)
    
    def verbose2(self,txt):
        if self.verbose >= 2:
            print(txt)
    
    def verbose3(self,txt):
        if self.verbose >= 3:
            print(txt)
    
    def verbose4(self,txt):
        if self.verbose >= 4:
            print(txt)
    
    def timed(self,txt,verbosity):
        laps=time.perf_counter()
        elapsed=laps-self.intermediatetime
        self.intermediatetime=laps
        if self.verbose >= verbosity:
            print("\n")
            print(txt + f" took {elapsed:.2f} seconds")
            print("\n")
    
    def sigma_mo_gabi(self,i,k,o,m):
        # return self.eri_mo_gabi[i,k,o,m]-self.eri_mo_gabi[i,k,m,o]
        return self.eri_mo_gabi[i,k,o,m]
    
    def sigma_mo_gabi_W(self,i,k,o,m):
        # return self.eri_mo_gabi[i,k,o,m]-self.eri_mo_gabi[i,k,m,o]
        return self.eri_mo_gabi_W[i,k,o,m]

    # for sparse use single basis
    def virtualShift(self,index):
        if index<self.nO:
            return 0
        return self.shift_virtual_energy
    
    def sqrt(self,x):
        return np.sqrt(x).astype(self.data_type_sparse)
#%% ordering the BSE kernels and creating the G03 space    
    
    def vbar(self):
        vout=np.zeros((self.nBas*2,self.nBas*2,self.nBas*2,self.nBas*2))
        self.verbose4("Vout calculation")
        for spinmu in range(2):
            for spinnu in range(2):
                for spinla in range(2):
                    for spinsi in range(2):
                        for mu in range(self.nBas):
                            for nu in range(self.nBas):
                                for la in range(self.nBas):
                                    for si in range(self.nBas):
                                        mu2 = (mu) * 2 
                                        nu2 = (nu) * 2 
                                        la2 = (la) * 2 
                                        si2 = (si) * 2 
                                        
                                        mu2 += 1 if (spinmu == 1) else 0
                                        nu2 += 1 if (spinnu == 1) else 0
                                        la2 += 1 if (spinla == 1) else 0
                                        si2 += 1 if (spinsi == 1) else 0
                                        
                                        spinmatch1 = 0.0
                                        spinmatch2 = 0.0
                                        spintotalmatch = 0.0
                                        
                                        if ((spinmu==spinnu) or (spinla == spinsi)):
                                            spinmatch1 = 1.0
                                        if ((spinmu==spinsi) or (spinla == spinnu)):
                                            spinmatch2 = 1.0
                                        
                                        spinsum = spinmu + spinnu + spinla + spinsi
                                        if (spinsum % 2 == 0):
                                            spintotalmatch = 1.0
                                            
                                        #chemists: 1234 -> physi 1324 -> gabi 1342
                                        vout[mu2,la2,si2,nu2]=spintotalmatch * (self.eri_mo_gabi[mu,la,si,nu]*spinmatch1 - self.eri_mo_gabi[mu,la,nu,si]*spinmatch2)
                                        
                                        if (abs(vout[mu2,la2,si2,nu2])>1e-5 and self.verbose>=4):
                                            self.verbose4('%4d %4d %4d %4d      %.5f'%(mu2,la2,si2,nu2,vout[mu2,la2,si2,nu2]))
        return vout
    
    
    def vbarOnTheSpot(self,mu2,la2,si2,nu2):
        mu=mu2//2
        la=la2//2
        nu=nu2//2
        si=si2//2
        
        
        spinmu=mu2%2
        spinla=la2%2
        spinnu=nu2%2
        spinsi=si2%2
        
        spinmatch1 = 0.0
        spinmatch2 = 0.0
        spintotalmatch = 0.0
        
        if ((spinmu==spinnu) or (spinla == spinsi)):
            spinmatch1 = 1.0
        if ((spinmu==spinsi) or (spinla == spinnu)):
            spinmatch2 = 1.0
        
        spinsum = spinmu + spinnu + spinla + spinsi
        if (spinsum % 2 == 0):
            spintotalmatch = 1.0
        
        return spintotalmatch * (self.eri_mo_gabi[mu,la,si,nu]*spinmatch1 - self.eri_mo_gabi[mu,la,nu,si]*spinmatch2)
    
    def g03spaceSelection(self):
        self.verbose4("G03 space selection")
        long_ab=np.arange(0,self.nBas*2)
        # rule ijl; i>j and l is occupied MO (electron that needs to be removed)
        space=[]
        for i in long_ab:
            fi=1 if i < self.nO*2 else 0
            for j in long_ab:
                fj=1 if j < self.nO*2 else 0
                if i>j:
                    for l in long_ab: 
                        fl=1 if l < self.nO*2 else 0
                        if (fi-fl)*(fj-fl) != 0:
                            space.append([i,j,l])
                            self.verbose4('%4d %4d %4d'%(i,j,l))
        return space              

#%% For the full effective Hamiltonian, creating the self energy shoulder and the self energy body

    def g3sigma(self):
        self.verbose4("The Sigma3 body")
        sigma3matrix=np.zeros((self.nBas*2,self.nBas*2,self.nBas*2,self.nBas*2,self.nBas*2,self.nBas*2))
        nspace=self.nspace
        for idx in range(len(nspace)):
            [i,j,l] = nspace[idx]
            
            fi = 0 if (i >= self.nO*2) else 1
            fj = 0 if (j >= self.nO*2) else 1
            fl = 0 if (l >= self.nO*2) else 1
            
            for jdx in range(len(nspace)):
                [m,o,k]=nspace[jdx]
                
                dlk = 1 if (l==k) else 0
                dmj = 1 if (m==j) else 0
                dio = 1 if (i==o) else 0
                doj = 1 if (o==j) else 0
                dim = 1 if (i==m) else 0
                
                if ((fi-fl)*(fj-fl)==0):
                    continue
                
                prefac=((1-fi)*(1-fj)*fl-fi*fj*(1-fl))
                lkterm=dlk*self.vout[i,j,o,m]
                mjterm=dmj*self.vout[i,k,l,o]
                ioterm=dio*self.vout[j,k,l,m]
                ojterm=doj*self.vout[i,k,l,m]
                imterm=dim*self.vout[j,k,l,o]
                    
                sigma3matrix[i, j, l, m, o, k]=prefac*(lkterm+mjterm+ioterm-ojterm-imterm)
                if (abs(sigma3matrix[i, j, l, m, o, k])>1e-15):
                    self.compare.append([str(i),str(j),str(l),str(m),str(o),str(k),f"{sigma3matrix[i, j, l, m, o, k]:.4f}"])
                    self.verbose4('%4d %4d %4d %4d %4d %4d      %.5f'%(i,j,l,m,o,k,sigma3matrix[i, j, l, m, o, k]))
                    
        return sigma3matrix
    
    def sigma3shoulder(self):
        self.verbose4("The Sigma3 shoulder")
        sigma3s=np.zeros((self.nBas*2,self.nBas*2,self.nBas*2,self.nBas*2))
        nspace=self.nspace
        for i in range(self.nBas*2):
            for idx in range(len(nspace)):
                [m,o,k] = nspace[idx]
                sigma3s[i, m, o, k]=self.vout[i,k,o,m]
                self.verbose4('%4d %4d %4d %4d      %.5f'%(i,m,o,k,sigma3s[i, m, o, k]))
                if (abs(sigma3s[i, m, o, k])>1e-15):
                    self.compareshoulder.append([str(i),str(m),str(o),str(k),f"{sigma3s[i, m, o, k]:.4f}"])
                    self.verbose4('%4d %4d %4d %4d      %.5f'%(i,m,o,k,sigma3s[i, m, o, k]))
        return sigma3s
    
#%% creating the full effective Hamiltonians

    def createSecondBornEffectiveHam(self):
        self.verbose4("H effective")
        e01energies=np.zeros((self.nBas*2))
        self.verbose4("writing the double basis hf energies")
        for idx in range(self.nBas*2):
            jdx=int((idx)/2)
            e01energies[idx]=self.mo_en[jdx]
            self.verbose4('%4d   %.5f'%(jdx,self.mo_en[jdx]))
        
        self.verbose4("writing the triple particle hf energies")
        e03energies=[]
        sigmaShoulder=np.zeros((self.nBas*2,len(self.nspace)))
        sigmaBody=np.zeros((len(self.nspace),len(self.nspace)))
        for index,element in enumerate(self.nspace):
            a=element[0]
            b=element[1]
            c=element[2]
            
            e03energies.append(e01energies[a]+e01energies[b]-e01energies[c])
            self.verbose4('%4d %4d %4d   %.5f'%(a,b,c,e03energies[index]))
            
            
            for jdx in range(self.nBas*2):
                sigmaShoulder[jdx,index]=self.sigma3s[jdx,a,b,c]
            
        h03=np.diag(np.concatenate((e01energies,e03energies)))
        
            
        
        selfie=np.block([
            [np.zeros((self.nBas*2,self.nBas*2)),sigmaShoulder],
            [sigmaShoulder.T,sigmaBody]
            ])
        
        self.heffZ=np.add(h03,selfie)
        self.verbose4("Nonzero values of effective Hamiltionian")
        if self.verbose>=4:
            for ii in range(len(self.heffZ)):
                for jj in range(len(self.heffZ)):
                    if (abs(self.heffZ[ii,jj])>1e-15):
                        self.verbose4('%4d %4d      %.5f'%(ii,jj,self.heffZ[ii,jj]))
        
        self.e03energies=e03energies
        self.e01energies=e01energies
        
        return np.add(h03,selfie)
        
    
    def HeffMCDE(self):
        if self.HeffSBmat is None:
            raise ValueError("The second Born effective Hamiltonian has not been initialised with createSecondBornEffectiveHam")
        if self.sigma3 is None:
            raise ValueError("3-particle self energy not initialized with sigma3s")
        
        index=self.nBas*2
        self.HeffMCDEmat=self.HeffSBmat.copy()
        sigmaBody=np.zeros((len(self.nspace),len(self.nspace)))
        for index,element in enumerate(self.nspace):
            a=element[0]
            b=element[1]
            c=element[2]
            for index2,element2 in enumerate(self.nspace):
                x=element2[0]
                y=element2[1]
                z=element2[2]
                sigmaBody[index,index2]=self.sigma3[a,b,c,x,y,z]
        # print("Target shape:", self.HeffMCDEmat[index:index+sigmaBody.shape[0],
        #                                 index:index+sigmaBody.shape[1]].shape)
        # print("sigmaBody shape:", sigmaBody.shape)
        index=self.nBas*2
        self.HeffMCDEmat[index:index+sigmaBody.shape[0], index:index+sigmaBody.shape[1]]+=sigmaBody
        
        self.verbose4("Nonzero values of effective Hamiltionian")
        if self.verbose>=4:
            for ii in range(len(self.HeffMCDEmat)):
                for jj in range(len(self.HeffMCDEmat)):
                    if (abs(self.HeffMCDEmat[ii,jj])>1e-15):
                        self.verbose4('%4d %4d      %.5f'%(ii,jj,self.HeffMCDEmat[ii,jj]))
        
    
    def HeffSecondBorn(self):
        if self.HeffSBmat is None:
            raise ValueError("The second Born effective Hamiltonian has not been initialised with createSecondBornEffectiveHam")

        
        
        
        self.verbose4("Nonzero values of effective Hamiltionian")
        if self.verbose>=4:
            for ii in range(len(self.heffSBmat)):
                for jj in range(len(self.heffSBmat)):
                    if (abs(self.heffSBmat[ii,jj])>1e-15):
                        self.verbose4('%4d %4d      %.5f'%(ii,jj,self.heffSBmat[ii,jj]))
    
    def OldHeffSB(self):
        self.verbose4("H effective")
        e01energies=np.zeros((self.nBas*2))
        self.verbose4("writing the double basis hf energies")
        for idx in range(self.nBas*2):
            jdx=int((idx)/2)
            e01energies[idx]=self.mo_en[jdx]
            self.verbose4('%4d   %.5f'%(jdx,self.mo_en[jdx]))
        
        self.verbose4("writing the triple particle hf energies")
        e03energies=[]
        sigmaShoulder=np.zeros((self.nBas*2,len(self.nspace)))
        sigmaBody=np.zeros((len(self.nspace),len(self.nspace)))
        for index,element in enumerate(self.nspace):
            a=element[0]
            b=element[1]
            c=element[2]
            
            e03energies.append(e01energies[a]+e01energies[b]-e01energies[c])
            self.verbose4('%4d %4d %4d   %.5f'%(a,b,c,e03energies[index]))
            
            
            for jdx in range(self.nBas*2):
                sigmaShoulder[jdx,index]=self.sigma3s[jdx,a,b,c]
            
        
            
        
        h03=np.diag(np.concatenate((e01energies,e03energies)))
        
            
        
        selfie=np.block([
            [np.zeros((self.nBas*2,self.nBas*2)),sigmaShoulder],
            [sigmaShoulder.T,sigmaBody]
            ])
        
        self.heffZ=np.add(h03,selfie)
        self.verbose4("Nonzero values of effective Hamiltionian")
        if self.verbose>=4:
            for ii in range(len(self.heffZ)):
                for jj in range(len(self.heffZ)):
                    if (abs(self.heffZ[ii,jj])>1e-15):
                        self.verbose4('%4d %4d      %.5f'%(ii,jj,self.heffZ[ii,jj]))
        
        self.HeffSBmat=np.add(h03,selfie)
        evals,evecs=self.AuxillaryFunctions.eig(np.add(h03,selfie))
        

        self.verbose4("Eigenvalues for H effective")
        if self.verbose>=4:
            idx = np.argsort(evals)  # use -eigvals for descending
            eigvals_sorted = evals[idx]
            eigvecs_sorted = evecs[:, idx]
            for ii in range(len(evals)):
                self.verbose4('%4d      %.5f'%(ii,eigvals_sorted[ii]))
        
        return evals,evecs

    def OldHeff(self):
        self.verbose4("H effective")
        e01energies=np.zeros((self.nBas*2))
        self.verbose4("writing the double basis hf energies")
        for idx in range(self.nBas*2):
            jdx=int((idx)/2)
            e01energies[idx]=self.mo_en[jdx]
            self.verbose4('%4d   %.5f'%(jdx,self.mo_en[jdx]))
        
        self.verbose4("writing the triple particle hf energies")
        e03energies=[]
        sigmaShoulder=np.zeros((self.nBas*2,len(self.nspace)))
        sigmaBody=np.zeros((len(self.nspace),len(self.nspace)))
        for index,element in enumerate(self.nspace):
            a=element[0]
            b=element[1]
            c=element[2]
            
            e03energies.append(e01energies[a]+e01energies[b]-e01energies[c])
            self.verbose4('%4d %4d %4d   %.5f'%(a,b,c,e03energies[index]))
            
            
            for jdx in range(self.nBas*2):
                sigmaShoulder[jdx,index]=self.sigma3s[jdx,a,b,c]
            
            for index2,element2 in enumerate(self.nspace):
                x=element2[0]
                y=element2[1]
                z=element2[2]
                sigmaBody[index,index2]=self.sigma3[a,b,c,x,y,z]
            
        
        h03=np.diag(np.concatenate((e01energies,e03energies)))
        
            
        
        selfie=np.block([
            [np.zeros((self.nBas*2,self.nBas*2)),sigmaShoulder],
            [sigmaShoulder.T,sigmaBody]
            ])
        
        self.heffZ=np.add(h03,selfie)
        self.verbose4("Nonzero values of effective Hamiltionian")
        if self.verbose>=4:
            for ii in range(len(self.heffZ)):
                for jj in range(len(self.heffZ)):
                    if (abs(self.heffZ[ii,jj])>1e-15):
                        self.verbose4('%4d %4d      %.5f'%(ii,jj,self.heffZ[ii,jj]))
        
        evals,evecs=self.AuxillaryFunctions.eig(np.add(h03,selfie))
        idx = np.argsort(evals)  # use -eigvals for descending
       # eigvals_sorted = evals[idx]
       # eigvecs_sorted = evecs[:, idx]

        self.verbose4("Eigenvalues for H effective")
        if self.verbose>=4:
            for ii in range(len(evals)):
                self.verbose4('%4d      %.5f'%(ii,evals[ii]))
        
        return evals,evecs,sigmaShoulder,sigmaBody

#%% spin adapted effective Hamiltonian

    # def createSecondBornEffectiveSpinAdaptedHam(self):
        
    #     self.verbose3("create Second Born Effective Spin Adapted Hamiltonian")
        
        
    #     def d(a,b):
    #         return 1 if a==b else 0
        
        
    #     #transform spinorbitals to spatial orbitals
    #     nspace_spatials0=[]
    #     for entry in self.nspace:
    #         nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
    #     #remove duplicates
    #     seen = set()
    #     nspace_spatials = []
    #     for item in nspace_spatials0:
    #         t = tuple(item)
    #         if t not in seen:
    #             seen.add(t)
    #             nspace_spatials.append(item)
        
    #     self.verbose3("Length of 3-particle basis: " + str(len(nspace_spatials)))
    #     self.verbose4(nspace_spatials)
    #     ray=[]
    #     for i in range(self.nBas):
    #         ray.append(self.mo_en[i])
    #     head=np.diag(ray)
        
    #     #create wing
        
    #     wing=np.zeros((len(nspace_spatials)*2,len(ray)))
        
    #     for leftindex,left in enumerate(nspace_spatials):
    #         [i,j,l]=left
    #         for rightindex,m in enumerate(np.arange(self.nBas)):
    #             C3=np.sqrt(.5)**(d(i,j))*np.sqrt(.5)*(self.sigma_mo_gabi(i,j,l,m)+self.sigma_mo_gabi(i,j,m,l))
    #             C4=np.sqrt(3/2)*(self.sigma_mo_gabi(i,j,l,m)-self.sigma_mo_gabi(i,j,m,l))
                
    #             wing[2*leftindex,rightindex]=C3
    #             wing[2*leftindex+1,rightindex]=C4
        
    #     #create body matrix
        
    #     body=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
    #     # bodyHF=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
    #     for leftindex,left in enumerate(nspace_spatials):
    #         [i,j,l]=left
            
    #         for rightindex,right in enumerate(nspace_spatials):
    #             [m,o,k]=right
                
                
    #             A1=np.sqrt(.5)**(d(m,o))*np.sqrt(.5)**(d(i,j))*(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)+self.sigma_mo_gabi(i,j,m,o))
    #                                                           +d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-.5*self.sigma_mo_gabi(i,k,o,l))
    #                                                           +d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-.5*self.sigma_mo_gabi(j,k,m,l))
    #                                                           +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-.5*self.sigma_mo_gabi(i,k,m,l))
    #                                                           +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-.5*self.sigma_mo_gabi(j,k,o,l)))
    #             A2=(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)-self.sigma_mo_gabi(i,j,m,o))
    #                                                           -d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-1.5*self.sigma_mo_gabi(i,k,o,l))
    #                                                           -d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-1.5*self.sigma_mo_gabi(j,k,m,l))
    #                                                           +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-1.5*self.sigma_mo_gabi(i,k,m,l))
    #                                                           +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-1.5*self.sigma_mo_gabi(j,k,o,l)))
    #             F1=np.sqrt(.5)**(d(i,j))*(np.sqrt(3)/2)*(-d(m,j)*self.sigma_mo_gabi(i,k,o,l)+d(i,o)*self.sigma_mo_gabi(j,k,m,l)
    #                                                   +d(o,j)*self.sigma_mo_gabi(i,k,m,l)
    #                                                   -d(i,m)*self.sigma_mo_gabi(j,k,o,l))
    #             F2=np.sqrt(.5)**(d(m,o))*(np.sqrt(3)/2)*(d(m,j)*self.sigma_mo_gabi(i,k,o,l)-d(i,o)*self.sigma_mo_gabi(j,k,m,l)
    #                                                   +d(o,j)*self.sigma_mo_gabi(i,k,m,l)
    #                                                   -d(i,m)*self.sigma_mo_gabi(j,k,o,l))
                
                
    #             fi = 0 if (i >= self.nO) else 1
    #             fj = 0 if (j >= self.nO) else 1
    #             fl = 0 if (l >= self.nO) else 1
    #             prefac=-((1-fi)*(1-fj)*fl-fi*fj*(1-fl))
                
    #             ei=self.mo_en[i]
    #             ej=self.mo_en[j]
    #             el=self.mo_en[l]
    #             de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                
                
    #             body[2*leftindex,2*rightindex]=de+prefac*A1
    #             body[2*leftindex,2*rightindex+1]=prefac*F1
    #             body[2*leftindex+1,2*rightindex]=prefac*F2
    #             body[2*leftindex+1,2*rightindex+1]=de+prefac*A2
                
    #     H3upd=np.block([[head,np.transpose(wing)],[wing,body]])
        
    #     return H3upd
    
    # def HeffSBSpinAdapt(self):
    #     if self.HeffSBSpinAdaptMat is None:
    #         raise ValueError("the effective Hamiltonian spin adapted matrix is not initialized with createSecondBornEffectiveSpinAdaptedHam")
    #     self.HeffSBSpinAdaptMat=np.where(np.abs(self.HeffSBSpinAdaptMat) < self.zero_tol, 0.0, self.HeffSBSpinAdaptMat)
    #     self.HeffSBSpinAdaptMat=self.AuxillaryFunctions.remove_isolated_diagonals(self.HeffSBSpinAdaptMat)
        
    # def HeffMCDESpinAdapt(self):
    #     if self.HeffSBSpinAdaptMat is None:
    #         raise ValueError("the effective Hamiltonian spin adapted matrix is not initialized with createSecondBornEffectiveSpinAdaptedHam")
    #     self.HeffMCDESpinAdapt=self.HeffSBSpinAdapt.copy()
        
        
        
    #     self.HeffMCDESpinAdaptMat=np.where(np.abs(self.HeffMCDESpinAdaptMat) < self.zero_tol, 0.0, self.HeffMCDESpinAdaptMat)
    #     self.HeffMCDESpinAdaptMat=self.AuxillaryFunctions.remove_isolated_diagonals(self.HeffMCDESpinAdaptMat)



    def spinAdaptedMCDE(self,secondBorn=False):
        
        self.verbose3("Spin transformed MCDE")
        self.verbose3("Second Born? "+str(secondBorn))
        
        def d(a,b):
            return 1 if a==b else 0
        
        
        #transform spinorbitals to spatial orbitals
        nspace_spatials0=[]
        for entry in self.nspace:
            nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
        #remove duplicates
        seen = set()
        nspace_spatials = []
        for item in nspace_spatials0:
            t = tuple(item)
            if t not in seen:
                seen.add(t)
                nspace_spatials.append(item)
        
        #initialize 1part+3part nspace index
        spaceObj=MCDE.Nspace(np.arange(self.nBas),np.repeat(nspace_spatials,2,axis=0))
        
        self.verbose3("Length of 3-particle basis: " + str(len(nspace_spatials)))
        self.verbose4(nspace_spatials)
        ray=[]
        for i in range(self.nBas):
            ray.append(self.mo_en[i])
        head=np.diag(ray)
        
        #create wing
        
        wing=np.zeros((len(nspace_spatials)*2,len(ray)))
        
        for leftindex,left in enumerate(nspace_spatials):
            [i,j,l]=left
            for rightindex,m in enumerate(np.arange(self.nBas)):
                C3=np.sqrt(.5)**(d(i,j))*np.sqrt(.5)*(self.sigma_mo_gabi(i,j,l,m)+self.sigma_mo_gabi(i,j,m,l))
                C4=np.sqrt(3/2)*(self.sigma_mo_gabi(i,j,l,m)-self.sigma_mo_gabi(i,j,m,l))
                
                wing[2*leftindex,rightindex]=C3
                wing[2*leftindex+1,rightindex]=C4
        
        #create body matrix
        
        body=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
        # bodyHF=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
        for leftindex,left in enumerate(nspace_spatials):
            [i,j,l]=left
            
            for rightindex,right in enumerate(nspace_spatials):
                [m,o,k]=right
                
                
                A1=np.sqrt(.5)**(d(m,o))*np.sqrt(.5)**(d(i,j))*(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)+self.sigma_mo_gabi(i,j,m,o))
                                                              +d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-.5*self.sigma_mo_gabi(i,k,o,l))
                                                              +d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-.5*self.sigma_mo_gabi(j,k,m,l))
                                                              +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-.5*self.sigma_mo_gabi(i,k,m,l))
                                                              +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-.5*self.sigma_mo_gabi(j,k,o,l)))
                A2=(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)-self.sigma_mo_gabi(i,j,m,o))
                                                              -d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-1.5*self.sigma_mo_gabi(i,k,o,l))
                                                              -d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-1.5*self.sigma_mo_gabi(j,k,m,l))
                                                              +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-1.5*self.sigma_mo_gabi(i,k,m,l))
                                                              +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-1.5*self.sigma_mo_gabi(j,k,o,l)))
                F1=np.sqrt(.5)**(d(i,j))*(np.sqrt(3)/2)*(-d(m,j)*self.sigma_mo_gabi(i,k,o,l)+d(i,o)*self.sigma_mo_gabi(j,k,m,l)
                                                      +d(o,j)*self.sigma_mo_gabi(i,k,m,l)
                                                      -d(i,m)*self.sigma_mo_gabi(j,k,o,l))
                F2=np.sqrt(.5)**(d(m,o))*(np.sqrt(3)/2)*(d(m,j)*self.sigma_mo_gabi(i,k,o,l)-d(i,o)*self.sigma_mo_gabi(j,k,m,l)
                                                      +d(o,j)*self.sigma_mo_gabi(i,k,m,l)
                                                      -d(i,m)*self.sigma_mo_gabi(j,k,o,l))
                
                
                fi = 0 if (i >= self.nO) else 1
                fj = 0 if (j >= self.nO) else 1
                fl = 0 if (l >= self.nO) else 1
                prefac=-((1-fi)*(1-fj)*fl-fi*fj*(1-fl))
                
                ei=self.mo_en[i]
                ej=self.mo_en[j]
                el=self.mo_en[l]
                de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                
                
                body[2*leftindex,2*rightindex]=de+prefac*A1
                body[2*leftindex,2*rightindex+1]=prefac*F1
                body[2*leftindex+1,2*rightindex]=prefac*F2
                body[2*leftindex+1,2*rightindex+1]=de+prefac*A2
                
                # bodyHF[2*leftindex,2*rightindex]=de
                # bodyHF[2*leftindex+1,2*rightindex+1]=de
        
        if secondBorn:
            bodySB=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            
            for leftindex,left in enumerate(nspace_spatials):
                [i,j,l]=left
                
                for rightindex,right in enumerate(nspace_spatials):
                    [m,o,k]=right
                    
                    
                    ei=self.mo_en[i]
                    ej=self.mo_en[j]
                    el=self.mo_en[l]
                    de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                    
                    
                    bodySB[2*leftindex,2*rightindex]=de
                    bodySB[2*leftindex,2*rightindex+1]=0
                    bodySB[2*leftindex+1,2*rightindex]=0
                    bodySB[2*leftindex+1,2*rightindex+1]=de
        
        H3upd=np.block([[head,np.transpose(wing)],[wing,body]])
        
        
        
        H3upd=np.where(np.abs(H3upd) < self.zero_tol, 0.0, H3upd)
        
        if secondBorn:
            H3SB=np.block([[head,np.transpose(wing)],[wing,bodySB]])
            H3SB=np.where(np.abs(H3SB) < self.zero_tol, 0.0, H3SB)
            H3SB,self.nspace_spatials_SB=self.AuxillaryFunctions.remove_isolated_diagonals(H3SB,spaceObj,self.remove_single_values)
        
        
        H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals(H3upd,spaceObj,self.remove_single_values)
        
        
        
        self.timed("Creating spin Opt eff. Hamiltonian",2)
        
        if secondBorn:
            return H3upd,H3SB
        return H3upd

    def spinAdaptedMCDEFullChunks(self,secondBorn=False):
        self.verbose3("Full MCDE Chunks")
        self.verbose3("Second Born? "+str(secondBorn))
        
        datatype=self.data_type_sparse
        NBAS=2*self.nBas
        NO=2*self.nO
        #  small number
        def sqrt(x):
            return np.sqrt(x).astype(datatype)
        
        def check32(arr):
            prod=np.prod(arr.shape)
            res=(prod*4==arr.nbytes)
            if res:
                print("Is 32")
            else:
                print("Is not 32, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*4)
                sys.exit()
                
        def check16(arr):
            return None
            prod=np.prod(arr.shape)
            res=(prod*2==arr.nbytes)
            if res:
                print("Is 16")
            else:
                print("Is not 16, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*2)
                sys.exit()
        

        
        nspace3particle = np.array(self.nspace)   # shape (N,3)
        spaceObj=MCDE.Nspace(np.arange(NBAS),self.nspace)

        # head
        ray=[]
        for i in range(NBAS):
            ray.append(self.mo_en[i//2])
        head=np.diag(ray).astype(datatype)
        
        # check32(head)

        nBasspace=np.arange(NBAS)
        mwing = nBasspace[:,None]
        
        N = len(nspace3particle)
        
        chunk_size = max(N//100,2)
        chunk_size = 5000 if chunk_size > 5000 else chunk_size
        chunk_size = 1
        nchunks = [
            (i, i + len(nspace3particle[i:i+chunk_size]), nspace3particle[i:i+chunk_size])
            for i in range(0, len(nspace3particle), chunk_size)
        ]
        
        
        
        def speedUpCore(nspaceL,nspaceR):
            
            body_out = None
            
            i = nspaceL[:, 0]
            j = nspaceL[:, 1]
            l = nspaceL[:, 2]
            
            m = nspaceR[:, 0]
            o = nspaceR[:, 1]
            k = nspaceR[:, 2]
            
            iL = i[:, None]
            jL = j[:, None]
            lL = l[:, None]
            
            mR = m[None, :]
            oR = o[None, :]
            kR = k[None, :]
            
            dim = (iL == mR).astype(datatype)
            djo = (jL == oR).astype(datatype)
            dlk = (lL == kR).astype(datatype)
            
            dmj = (mR == jL).astype(datatype)
            dio = (iL == oR).astype(datatype)
            doj = (oR == jL).astype(datatype)
            
            ei = self.mo_en[i//2].astype(datatype)
            ej = self.mo_en[j//2].astype(datatype)
            el = self.mo_en[l//2].astype(datatype)
            
            eiL = ei[:, None]
            ejL = ej[:, None]
            elL = el[:, None]
            
            # check16(dlk)
            
            de = ((eiL - (elL - ejL)) * dim * djo * dlk).astype(datatype)
          
            # check32(de)
            #memory expensive!
            
            sigma=self.eri_mo_gabi.astype(datatype)
            
            
            # S_ijom = (iL%2 == mR%2 or jL%2 == oR%2)*sigma[iL, jL, oR, mR].astype(datatype)
            # S_ijmo = (iL%2 == oR%2 or jL%2 == mR%2)*sigma[iL, jL, mR, oR].astype(datatype)
            # S_iklo = (iL%2 == oR%2 or lL%2 == kR%2)*sigma[iL, kR, lL, oR].astype(datatype)
            # S_ikol = (iL%2 == lL%2 or oR%2 == kR%2)*sigma[iL, kR, oR, lL].astype(datatype)
            # S_jklm = (jL%2 == mR%2 or lL%2 == kR%2)*sigma[jL, kR, lL, mR].astype(datatype)
            # S_jkml = (jL%2 == lL%2 or mR%2 == kR%2)*sigma[jL, kR, mR, lL].astype(datatype)
            # S_iklm = (iL%2 == mR%2 or lL%2 == kR%2)*sigma[iL, kR, lL, mR].astype(datatype)
            # S_ikml = (iL%2 == lL%2 or mR%2 == kR%2)*sigma[iL, kR, mR, lL].astype(datatype)
            # S_jklo = (jL%2 == oR%2 or kR%2 == lL%2)*sigma[jL, kR, lL, oR].astype(datatype)
            # S_jkol = (jL%2 == lL%2 or kR%2 == oR%2)*sigma[jL, kR, oR, lL].astype(datatype)
            
            dividefactor=2
            S_1 = (((iL%2 + mR%2 + jL%2 + oR%2)%2==0)     
                   *(((iL % 2 == mR % 2) | (jL % 2 == oR % 2))
                                                   *sigma[iL//dividefactor, jL//dividefactor, oR//dividefactor, mR//dividefactor]-
                                                   ((iL % 2 == oR % 2) | (jL % 2 == mR % 2))*sigma[iL//dividefactor, jL//dividefactor, mR//dividefactor, oR//dividefactor])).astype(datatype)
            S_2 = (((iL%2 + kR%2 + lL%2 + oR%2)%2==0)
                   *(((iL % 2 == oR % 2) | (lL % 2 == kR % 2))*sigma[iL//dividefactor, kR//dividefactor, lL//dividefactor, oR//dividefactor]-
                                                   ((iL % 2 == kR % 2) | (lL % 2 == oR % 2))*sigma[iL//2, kR//dividefactor, oR//dividefactor, lL//dividefactor])).astype(datatype)
            S_3 = (((jL%2 + kR%2 + lL%2 + mR%2)%2==0)
                   *(((jL % 2 == mR % 2) | (lL % 2 == kR % 2))*sigma[jL//dividefactor, kR//dividefactor, lL//dividefactor, mR//dividefactor]-
                                                   ((jL % 2 == lL % 2) | (mR % 2 == kR % 2))*sigma[jL//dividefactor, kR//dividefactor, mR//dividefactor, lL//dividefactor])).astype(datatype)
            S_4 = (((iL%2 + kR%2 + lL%2 + mR%2)%2==0)
                   *(((iL % 2 == mR % 2) | (lL % 2 == kR % 2))*sigma[iL//dividefactor, kR//dividefactor, lL//dividefactor, mR//dividefactor]-
                                                   ((iL % 2 == lL % 2) | (mR % 2 == kR % 2))*sigma[iL//dividefactor, kR//dividefactor, mR//dividefactor, lL//dividefactor])).astype(datatype)
            S_5 = (((jL%2 + kR%2 + lL%2 + oR%2)%2==0)
                   *(((jL % 2 == oR % 2) | (lL % 2 == kR % 2))*sigma[jL//dividefactor, kR//dividefactor, lL//dividefactor, oR//dividefactor]-
                                                   ((jL % 2 == lL % 2) | (kR % 2 == oR % 2))*sigma[jL//dividefactor, kR//dividefactor, oR//dividefactor, lL//dividefactor])).astype(datatype)
            
            # spinmatch_ijom = (iL%2 == mR%2 or jL%2 == oR%2).astype(datatype)
            # spinmatch_ijmo = (iL%2 == oR%2 or jL%2 == mR%2).astype(datatype)
            # spinmatch_iklo = (iL%2 == oR%2 or lL%2 == kR%2).astype(datatype)
            # spinmatch_ikol = (iL%2 == lL%2 or oR%2 == kR%2).astype(datatype)
            # spinmatch_jklm = (jL%2 == mR%2 or lL%2 == kR%2).astype(datatype)
            # spinmatch_jkml = (jL%2 == lL%2 or mR%2 == kR%2).astype(datatype)
            # spinmatch_iklm = (iL%2 == mR%2 or lL%2 == kR%2).astype(datatype)
            # spinmatch_ikml = (iL%2 == lL%2 or mR%2 == kR%2).astype(datatype)
            # spinmatch_jklo = (jL%2 == oR%2 or kR%2 == lL%2).astype(datatype)
            # spinmatch_jkol = (jL%2 == lL%2 or kR%2 == oR%2).astype(datatype)
            
            # spintotalmatch_ijom=((iL%2 + mR%2 + jL%2 + oR%2)%2==0).astype(datatype)
            # spintotalmatch_iklo=((iL%2 + kR%2 + lL%2 + oR%2)%2==0).astype(datatype)
            # spintotalmatch_jklm=((jL%2 + kR%2 + lL%2 + mR%2)%2==0).astype(datatype)
            # spintotalmatch_iklm=((iL%2 + kR%2 + lL%2 + mR%2)%2==0).astype(datatype)
            # spintotalmatch_jklo=((jL%2 + kR%2 + lL%2 + oR%2)%2==0).astype(datatype)
            
            djo = (jL == oR).astype(datatype)
            dlk = (lL == kR).astype(datatype)
            
            dmj = (mR == jL).astype(datatype)
            dio = (iL == oR).astype(datatype)
            doj = (oR == jL).astype(datatype)
            
            fi = (i < NO).astype(np.int32)
            fj = (j < NO).astype(np.int32)
            fl = (l < NO).astype(np.int32)
            
            fiL = fi[:, None]
            fjL = fj[:, None]
            flL = fl[:, None]
            
            prefac = (((1-fiL)*(1-fjL)*flL - fiL*fjL*(1-flL))).astype(datatype)
            # print(de)
            # print(prefac)
            # interaction = (de+prefac*(dlk*S_1+dmj*S_2+dio*S_3-doj*S_4-dim*S_5)).astype(datatype)
            body_out = (de+prefac*(dlk*S_1+dmj*S_2+dio*S_3-doj*S_4-dim*S_5)).astype(datatype)
            
            # print("the body")
            # print(body_out)
            
            
            # print("the interaction")
            # print(interaction)
            # print(interaction - interaction.T)
            
            # terms = {
            #     "dlk*S1": dlk*S_1,
            #     "dmj*S2": dmj*S_2,
            #     "dio*S3": dio*S_3,
            #     "doj*S4": doj*S_4,
            #     "dim*S5": dim*S_5,
            # }
            
            # for name, term in terms.items():
            #     print(name)
            #     print(term)
            #     print(term - term.T)
            
            # sys.exit()
            # body = de
            return body_out
        
        def speedUpShoulder(nspaceR):

            m = nspaceR[:, 0]
            o = nspaceR[:, 1]
            k = nspaceR[:, 2]
            
            mR = m[None, :]
            oR = o[None, :]
            kR = k[None, :]
                    
            sigma=self.eri_mo_gabi.astype(datatype)
         
            dividefactor=2

            cond1 = ((mwing % 2 + mR % 2 + kR % 2 + oR % 2) % 2 == 0).astype(datatype)
            
            term1 = ((mwing % 2 == mR % 2) | (kR % 2 == oR % 2)).astype(datatype)
            
            term2 = ((mwing % 2 == oR % 2) | (kR % 2 == mR % 2)).astype(datatype)
            
            s_1 = (
                cond1 *
                (
                    term1 * sigma[mwing // dividefactor,
                                  kR // dividefactor,
                                  oR // dividefactor,
                                  mR // dividefactor]
                    -
                    term2 * sigma[mwing // dividefactor,
                                  kR // dividefactor,
                                  mR // dividefactor,
                                  oR // dividefactor]
                )
            ).astype(datatype)
            
            return s_1
        
        body = np.zeros((N, N),dtype=datatype)
          
        wing = np.zeros((NBAS,len(nspace3particle)),dtype=datatype)
        # print(wing.shape)
        for tchunkR in nchunks:
            (indexR0, indexR1, chunkR) = tchunkR
            shoulderchunk=speedUpShoulder(chunkR)
            c0 = indexR0
            c1 = indexR1
            # print(c0)
            # print(c1)
            # print(shoulderchunk.shape)
            # print(wing[:NBAS,c0:c1].shape)
            # print(wing[:NBAS,c0:c1].shape)
            wing[:NBAS,c0:c1] = shoulderchunk
            for tchunkL in nchunks:
                
                (indexL0, indexL1, chunkL) = tchunkL
                
                
                # indexL0=indexL
                # indexL1=indexL+chunk_size
                # indexR0=indexR
                # indexR1=indexR+chunk_size
                
                bodychunk=speedUpCore(chunkL,chunkR)
                bodychunkB=speedUpCore(chunkR,chunkL).conj().T
                if not np.allclose(bodychunk,bodychunkB):
                    print(bodychunk)
                    print(bodychunkB)
                    print(chunkL)
                    print(chunkR)
                    print(indexL0)
                    print(indexR0)
                    sys.exit()
                
                r0 = indexL0
                r1 = indexL1
                
                
                # even-even (A)
                body[r0:r1, c0:c1] = bodychunk

        body[np.abs(body) < self.sparse_tol] = 0.0
        head[np.abs(head) < self.sparse_tol] = 0.0
        wing[np.abs(wing) < self.sparse_tol] = 0.0
        # body_sparse = csr_matrix(body)
        # print(wing.shape)
        # print(body.shape)
        # print(head.shape)
        
        
        H3upd=scipy.sparse.bmat([[head,wing],[np.transpose(wing),body]], format='csr', dtype=datatype)
        
        # check16(H3upd)
        
        H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3upd,spaceObj,self.remove_single_values)
        
        self.timed("Creating spin Opt eff. Hamiltonian sparse",2)
        
        
        
        return H3upd

#%% Sparse
    
    @staticmethod
    def compute_body_block(args):
        (chunk, nspace_spatials, mo_en, nO, sparse_tol, zero_tol, sigma_func, datatype, sqrt) = args
    
        def check32(arr):
            prod=np.prod(arr.shape)
            res=(prod*4==arr.nbytes)
            if res:
                print("Is 32")
            else:
                print("Is not 32, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*4)
                sys.exit()
    
        d = lambda a,b: 1 if a==b else 0
    
        
        results = []
    
        
        
    
        prefacA1=datatype(0.5)
        prefacA2=datatype(1.5)
        sqrt2=sqrt(.5)
        sqrt3 = (sqrt(3.0) / 2.0).astype(datatype)
        
        
        for leftindex in chunk:
            i, j, l = nspace_spatials[leftindex]
            ei = mo_en[i]
            ej = mo_en[j]
            el = mo_en[l]
        
            fi = datatype(0) if (i >= nO) else datatype(1)
            fj = datatype(0) if (j >= nO) else datatype(1)
            fl = datatype(0) if (l >= nO) else datatype(1)
            prefac_base = datatype(-((1-fi)*(1-fj)*fl - fi*fj*(1-fl)))
            for rightindex, right in enumerate(nspace_spatials):
                m,o,k = right
        
                de = (ei - (el - ej)) * d(i,m)*d(j,o)*d(l,k)
        
                # A1
                
                s_ij=sqrt2**d(i,j)
                s_mo = sqrt2**d(m,o)
                
                A1 = s_mo * s_ij * (
                    -d(l,k) * (sigma_func[i,j,o,m] + sigma_func[i,j,m,o])
                    + d(m,j) * (sigma_func[i,k,l,o] - prefacA1 * sigma_func[i,k,o,l])
                    + d(i,o) * (sigma_func[j,k,l,m] - prefacA1 * sigma_func[j,k,m,l])
                    + d(o,j) * (sigma_func[i,k,l,m] - prefacA1 * sigma_func[i,k,m,l])
                    + d(i,m) * (sigma_func[j,k,l,o] - prefacA1 * sigma_func[j,k,o,l])
                )
                
                # A2
                A2 = (
                    -d(l,k)*(sigma_func[i,j,o,m]-sigma_func[i,j,m,o])
                    -d(m,j)*(sigma_func[i,k,l,o]-prefacA2*sigma_func[i,k,o,l])
                    -d(i,o)*(sigma_func[j,k,l,m]-prefacA2*sigma_func[j,k,m,l])
                    +d(o,j)*(sigma_func[i,k,l,m]-prefacA2*sigma_func[i,k,m,l])
                    +d(i,m)*(sigma_func[j,k,l,o]-prefacA2*sigma_func[i,k,o,l])
                )
        
                A2 = (
                    -d(l,k) * (sigma_func[i,j,o,m] - sigma_func[i,j,m,o])
                    - d(m,j) * (sigma_func[i,k,l,o] - prefacA2 * sigma_func[i,k,o,l])
                    - d(i,o) * (sigma_func[j,k,l,m] - prefacA2 * sigma_func[j,k,m,l])
                    + d(o,j) * (sigma_func[i,k,l,m] - prefacA2 * sigma_func[i,k,m,l])
                    + d(i,m) * (sigma_func[j,k,l,o] - prefacA2 * sigma_func[j,k,o,l])
                )
        
                F1 = np.sqrt(.5)**(d(i,j))*(np.sqrt(3)/2)*(
                    -d(m,j)*sigma_func[i,k,o,l]
                    +d(i,o)*sigma_func[j,k,m,l]
                    +d(o,j)*sigma_func[i,k,m,l]
                    -d(i,m)*sigma_func[j,k,o,l]
                )
        
                F1 = s_ij * sqrt3 * (
                    -d(m,j) * sigma_func[i,k,o,l]
                    + d(i,o) * sigma_func[j,k,m,l]
                    + d(o,j) * sigma_func[i,k,m,l]
                    - d(i,m) * sigma_func[j,k,o,l]
                )
                
                F2 = s_mo * sqrt3 * (
                    d(m,j) * sigma_func[i,k,o,l]
                    - d(i,o) * sigma_func[j,k,m,l]
                    + d(o,j) * sigma_func[i,k,m,l]
                    - d(i,m) * sigma_func[j,k,o,l]
                )
                
               
        
                A = de + prefac_base*A1
                B = prefac_base*F1
                C = prefac_base*F2
                D = de + prefac_base*A2
        
                if abs(A) > sparse_tol:
                    results.append((2*leftindex, 2*rightindex, A))
                if abs(B) > sparse_tol:
                    results.append((2*leftindex, 2*rightindex+1, B))
                if abs(C) > sparse_tol:
                    results.append((2*leftindex+1, 2*rightindex, C))
                if abs(D) > sparse_tol:
                    results.append((2*leftindex+1, 2*rightindex+1, D))
    
        return results
    
    @staticmethod
    def compute_wing_block(args):
        (chunk, nspace_spatials, nBas, mo_en, nO, sparse_tol, zero_tol, sigma_func, datatype, sqrt) = args
    
        d = lambda a,b: 1 if a==b else 0
    
        
        results = []
        
        sqrt2=sqrt(.5)
        pref4=sqrt(3/2)
        for leftindex in chunk:
            i, j, l = nspace_spatials[leftindex]
            for rightindex,m in enumerate(np.arange(nBas)):
                
                pref=((sqrt2**d(i,j))*sqrt2).astype(datatype)
                
                S1 = sigma_func[i, j, l, m]
                S2 = sigma_func[i, j, m, l]
                
              
                C3 = pref * (S1 + S2)
                C4 = pref4 * (S1 - S2)

                if abs(C3) > sparse_tol:
                    results.append((2*leftindex,rightindex,C3))
                if abs(C4) > sparse_tol:
                    results.append((2*leftindex+1,rightindex,C4))

        return results

    def spinAdaptedMCDESparseParallel(self,secondBorn=False):
        
        self.verbose3("Spin transformed MCDE sparse")
        self.verbose3("Second Born? "+str(secondBorn))
        
        datatype=self.data_type_sparse
        
        def sqrt(x):
            return np.sqrt(x).astype(datatype)
        
        #transform spinorbitals to spatial orbitals
        nspace_spatials0=[]
        for entry in self.nspace:
            nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
        #remove duplicates
        seen = set()
        nspace_spatials = []
        for item in nspace_spatials0:
            t = tuple(item)
            if t not in seen:
                seen.add(t)
                nspace_spatials.append(item)
        
        
        
        nspace = np.array(nspace_spatials)   # shape (N,3)
        
        spaceObj=MCDE.Nspace(np.arange(self.nBas),np.repeat(nspace_spatials,2,axis=0))
        
        self.verbose3("Length of 3-particle basis: " + str(len(nspace_spatials)))
        self.verbose4(nspace_spatials)
        
        sigma=self.eri_mo_gabi.astype(datatype)
        
        ray=[]
        for i in range(self.nBas):
            ray.append(self.mo_en[i])
        head=lil_matrix(np.diag(ray))
        
        # create chunks
        n_workers = os.cpu_count()
        n = len(nspace_spatials)
        chunk_size = (n + n_workers - 1) // n_workers  # ceil division

        chunks = [
            list(range(i, min(i + chunk_size, n)))
            for i in range(0, n, chunk_size)
        ]
        
        tasks = [
            (chunk, nspace_spatials, self.mo_en.astype(datatype), self.nO,
             self.sparse_tol, self.zero_tol, sigma, datatype, self.sqrt)
            for chunk in chunks
        ]
        
        all_results = []
        
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(MCDE.compute_body_block, t) for t in tasks]
        
            for f in as_completed(futures):
                all_results.extend(f.result())
        
        body = lil_matrix((len(nspace_spatials)*2, 2*len(nspace_spatials)), dtype=datatype)

        for r, c, v in all_results:
            body[r, c] = v

        tasks = [
            (chunk, nspace_spatials, self.nBas, self.mo_en, self.nO,
             self.sparse_tol, self.zero_tol, sigma, datatype, self.sqrt)
            for chunk in chunks
        ]
        
        all_results = []
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(MCDE.compute_wing_block, t) for t in tasks]
        
            for f in as_completed(futures):
                all_results.extend(f.result())
        
        wing = lil_matrix((len(nspace_spatials)*2,len(ray)),dtype=datatype)
        
        for r, c, v in all_results:
            wing[r, c] = v
        
        H3upd = scipy.sparse.bmat([[head, wing.T],
              [wing, body]], format='csr',dtype=datatype)
        
        self.timed("Creating spin Opt eff. Hamiltonian sparse",2)
        
        H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3upd,spaceObj,self.remove_single_values)
     
        return H3upd

    def spinAdaptedMCDESparse(self,secondBorn=False):
        self.verbose3("Spin transformed MCDE sparse")
        self.verbose3("Second Born? "+str(secondBorn))
        
        datatype=self.data_type_sparse
        
        #  small number
        def sqrt(x):
            return np.sqrt(x).astype(datatype)
        
        def check32(arr):
            prod=np.prod(arr.shape)
            res=(prod*4==arr.nbytes)
            if res:
                print("Is 32")
            else:
                print("Is not 32, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*4)
                sys.exit()
                
        def check16(arr):
            return None
            prod=np.prod(arr.shape)
            res=(prod*2==arr.nbytes)
            if res:
                print("Is 16")
            else:
                print("Is not 16, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*2)
                sys.exit()
        
        #transform spinorbitals to spatial orbitals
        nspace_spatials0=[]
        for entry in self.nspace:
            nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
        #remove duplicates
        seen = set()
        nspace_spatials = []
        for item in nspace_spatials0:
            t = tuple(item)
            if t not in seen:
                seen.add(t)
                nspace_spatials.append(item)
        
        nspace = np.array(nspace_spatials)   # shape (N,3)
        spaceObj=MCDE.Nspace(np.arange(self.nBas),np.repeat(nspace_spatials,2,axis=0))

        # head
        ray=[]
        for i in range(self.nBas):
            ray.append(self.mo_en[i])
        head=np.diag(ray).astype(datatype)
        
        # check32(head)

        nBasspace=np.arange(self.nBas)
        mwing = nBasspace[None, :]

        i = nspace[:, 0]
        j = nspace[:, 1]
        l = nspace[:, 2]
        
        m = nspace[:, 0]
        o = nspace[:, 1]
        k = nspace[:, 2]
        
        iL = i[:, None]
        jL = j[:, None]
        lL = l[:, None]
        
        mR = m[None, :]
        oR = o[None, :]
        kR = k[None, :]
        
        dim = (iL == mR).astype(datatype)
        djo = (jL == oR).astype(datatype)
        dlk = (lL == kR).astype(datatype)
        
        dmj = (mR == jL).astype(datatype)
        dio = (iL == oR).astype(datatype)
        doj = (oR == jL).astype(datatype)
        
        ei = self.mo_en[i].astype(datatype)
        ej = self.mo_en[j].astype(datatype)
        el = self.mo_en[l].astype(datatype)
        
        eiL = ei[:, None]
        ejL = ej[:, None]
        elL = el[:, None]
        
        check16(dlk)
        
        de = ((eiL - (elL - ejL)) * dim * djo * dlk).astype(datatype)
      
        # check32(de)
        #memory expensive!
        N=self.nBas
        
        sigma=self.eri_mo_gabi.astype(datatype)
        
        
        
        S_ijom = sigma[iL, jL, oR, mR]
        S_ijmo = sigma[iL, jL, mR, oR]
        S_iklo = sigma[iL, kR, lL, oR]
        S_ikol = sigma[iL, kR, oR, lL]
        S_jklm = sigma[jL, kR, lL, mR]
        S_jkml = sigma[jL, kR, mR, lL]
        S_iklm = sigma[iL, kR, lL, mR]
        S_ikml = sigma[iL, kR, mR, lL]
        S_jklo = sigma[jL, kR, lL, oR]
        S_jkol = sigma[jL, kR, oR, lL]
        
        
        
        sqrt2 = sqrt(0.5)
        sqrt3 = (sqrt(3.0) / 2.0).astype(datatype)
        

        
        s_mo = np.where(mR != oR, 1.0, sqrt2).astype(datatype)
        s_ij = np.where(iL != jL, 1.0, sqrt2).astype(datatype)
        
        # check32(s_mo)
        
        prefacA1=datatype(0.5)
        prefacA2=datatype(1.5)
        
        A1 = s_mo * s_ij * (
            -dlk * (S_ijom + S_ijmo)
            + dmj * (S_iklo - prefacA1 * S_ikol)
            + dio * (S_jklm - prefacA1 * S_jkml)
            + doj * (S_iklm - prefacA1 * S_ikml)
            + dim * (S_jklo - prefacA1 * S_jkol)
        )
        check16(dmj * (S_iklo - prefacA1 * S_ikol))
        check16(A1)
        
        A2 = (
            -dlk * (S_ijom - S_ijmo)
            - dmj * (S_iklo - prefacA2 * S_ikol)
            - dio * (S_jklm - prefacA2 * S_jkml)
            + doj * (S_iklm - prefacA2 * S_ikml)
            + dim * (S_jklo - prefacA2 * S_jkol)
        )
        
        check16(A2)
        
        F1 = s_ij * sqrt3 * (
            -dmj * S_ikol
            + dio * S_jkml
            + doj * S_ikml
            - dim * S_jkol
        )
        #
        check16(F1)
        
        F2 = s_mo * sqrt3 * (
            dmj * S_ikol
            - dio * S_jkml
            + doj * S_ikml
            - dim * S_jkol
        )
        

        
        fi = (i < self.nO).astype(np.int32)
        fj = (j < self.nO).astype(np.int32)
        fl = (l < self.nO).astype(np.int32)
        
        fiL = fi[:, None]
        fjL = fj[:, None]
        flL = fl[:, None]
        
        prefac = (-((1-fiL)*(1-fjL)*flL - fiL*fjL*(1-flL))).astype(datatype)
        
        
        
        A = de + prefac * A1
        B = prefac * F1
        C = prefac * F2
        D = de + prefac * A2
        
  
        N = len(nspace)

        body = np.zeros((2*N, 2*N),dtype=datatype)
        
        body[0::2, 0::2] = A
        body[0::2, 1::2] = B
        body[1::2, 0::2] = C
        body[1::2, 1::2] = D
        
        #wing
        pref4=sqrt(3/2)
        pref=(np.where(iL==jL,sqrt2,1.0)*sqrt2).astype(datatype)
        
        S1 = sigma[iL, jL, lL, mwing]
        S2 = sigma[iL, jL, mwing, lL]
        
      
        C3 = pref * (S1 + S2)
        C4 = pref4 * (S1 - S2)
        
        
        
      
        wing = np.zeros((2*len(nspace), self.nBas),dtype=datatype)

        wing[0::2, :] = C3
        wing[1::2, :] = C4
        
        
        body[np.abs(body) < self.sparse_tol] = 0.0
        head[np.abs(head) < self.sparse_tol] = 0.0
        wing[np.abs(wing) < self.sparse_tol] = 0.0
        # body_sparse = csr_matrix(body)
        # print(wing.shape)
        # print(body.shape)
        # print(head.shape)
        
        
        H3upd=scipy.sparse.bmat([[head,np.transpose(wing)],[wing,body]], format='csr', dtype=datatype)
        
        # check16(H3upd)
        
        H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3upd,spaceObj,self.remove_single_values)
        
        self.timed("Creating spin Opt eff. Hamiltonian sparse",2)
        
        
        
        return H3upd
    
    

    def spinAdaptedMCDESparseChunks(self,secondBorn=False):
        self.verbose3("Spin transformed MCDE sparse")
        self.verbose3("Second Born? "+str(secondBorn))
        
        datatype=self.data_type_sparse
        
        #  small number
        def sqrt(x):
            return np.sqrt(x).astype(datatype)
        
        def check32(arr):
            prod=np.prod(arr.shape)
            res=(prod*4==arr.nbytes)
            if res:
                print("Is 32")
            else:
                print("Is not 32, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*4)
                sys.exit()
                
        def check16(arr):
            return None
            prod=np.prod(arr.shape)
            res=(prod*2==arr.nbytes)
            if res:
                print("Is 16")
            else:
                print("Is not 16, its ",type(arr))
                print("Size: ",arr.nbytes)
                print("Theory: ", prod*2)
                sys.exit()
        
        #transform spinorbitals to spatial orbitals
        nspace_spatials0=[]
        for entry in self.nspace:
            nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
        #remove duplicates
        seen = set()
        nspace_spatials = []
        for item in nspace_spatials0:
            t = tuple(item)
            if t not in seen:
                seen.add(t)
                nspace_spatials.append(item)
        
        nspaceFull = np.array(nspace_spatials)   # shape (N,3)
        spaceObj=MCDE.Nspace(np.arange(self.nBas),np.repeat(nspace_spatials,2,axis=0))

        # head
        ray=[]
        for i in range(self.nBas):
            ray.append(self.mo_en[i])
        head=np.diag(ray).astype(datatype)
        
        # check32(head)

        nBasspace=np.arange(self.nBas)
        mwing = nBasspace[None, :]
        
        N = len(nspaceFull)
        chunk_size = max(N//100,2)
        chunk_size = 5000 if chunk_size > 5000 else chunk_size
        
        nchunks = [
            (i, i + len(nspaceFull[i:i+chunk_size]), nspaceFull[i:i+chunk_size])
            for i in range(0, len(nspaceFull), chunk_size)
        ]
        
        body = np.zeros((2*N, 2*N),dtype=datatype)
          
        wing = np.zeros((2*len(nspaceFull), self.nBas),dtype=datatype)
        
        def speedUpCore(nspaceL,nspaceR):
            
            i = nspaceL[:, 0]
            j = nspaceL[:, 1]
            l = nspaceL[:, 2]
            
            m = nspaceR[:, 0]
            o = nspaceR[:, 1]
            k = nspaceR[:, 2]
            
            iL = i[:, None]
            jL = j[:, None]
            lL = l[:, None]
            
            mR = m[None, :]
            oR = o[None, :]
            kR = k[None, :]
            
            dim = (iL == mR).astype(datatype)
            djo = (jL == oR).astype(datatype)
            dlk = (lL == kR).astype(datatype)
            
            dmj = (mR == jL).astype(datatype)
            dio = (iL == oR).astype(datatype)
            doj = (oR == jL).astype(datatype)
            
            ei = self.mo_en[i].astype(datatype)
            ej = self.mo_en[j].astype(datatype)
            el = self.mo_en[l].astype(datatype)
            
            eiL = ei[:, None]
            ejL = ej[:, None]
            elL = el[:, None]
            
            # check16(dlk)
            
            de = ((eiL - (elL - ejL)) * dim * djo * dlk).astype(datatype)
          
            # check32(de)
            #memory expensive!
            
            sigma=self.eri_mo_gabi.astype(datatype)

            
            S_ijom = sigma[iL, jL, oR, mR]
            S_ijmo = sigma[iL, jL, mR, oR]
            S_iklo = sigma[iL, kR, lL, oR]
            S_ikol = sigma[iL, kR, oR, lL]
            S_jklm = sigma[jL, kR, lL, mR]
            S_jkml = sigma[jL, kR, mR, lL]
            S_iklm = sigma[iL, kR, lL, mR]
            S_ikml = sigma[iL, kR, mR, lL]
            S_jklo = sigma[jL, kR, lL, oR]
            S_jkol = sigma[jL, kR, oR, lL]
            
            
            
            sqrt2 = sqrt(0.5)
            sqrt3 = (sqrt(3.0) / 2.0).astype(datatype)
            

            
            s_mo = np.where(mR != oR, 1.0, sqrt2).astype(datatype)
            s_ij = np.where(iL != jL, 1.0, sqrt2).astype(datatype)
            
            # check32(s_mo)
            
            prefacA1=datatype(0.5)
            prefacA2=datatype(1.5)
            
            A1 = s_mo * s_ij * (
                -dlk * (S_ijom + S_ijmo)
                + dmj * (S_iklo - prefacA1 * S_ikol)
                + dio * (S_jklm - prefacA1 * S_jkml)
                + doj * (S_iklm - prefacA1 * S_ikml)
                + dim * (S_jklo - prefacA1 * S_jkol)
            )
            # check16(dmj * (S_iklo - prefacA1 * S_ikol))
            # check16(A1)
            
            A2 = (
                -dlk * (S_ijom - S_ijmo)
                - dmj * (S_iklo - prefacA2 * S_ikol)
                - dio * (S_jklm - prefacA2 * S_jkml)
                + doj * (S_iklm - prefacA2 * S_ikml)
                + dim * (S_jklo - prefacA2 * S_jkol)
            )
            
            # check16(A2)
            
            F1 = s_ij * sqrt3 * (
                -dmj * S_ikol
                + dio * S_jkml
                + doj * S_ikml
                - dim * S_jkol
            )
            #
            # check16(F1)
            
            F2 = s_mo * sqrt3 * (
                dmj * S_ikol
                - dio * S_jkml
                + doj * S_ikml
                - dim * S_jkol
            )
            
            fi = (i < self.nO).astype(np.int32)
            fj = (j < self.nO).astype(np.int32)
            fl = (l < self.nO).astype(np.int32)
            
            fiL = fi[:, None]
            fjL = fj[:, None]
            flL = fl[:, None]
            
            prefac = (-((1-fiL)*(1-fjL)*flL - fiL*fjL*(1-flL))).astype(datatype)

            A = de + prefac * A1
            B = prefac * F1
            C = prefac * F2
            D = de + prefac * A2
            
            pref4=sqrt(3/2)
            pref=(np.where(iL==jL,sqrt2,1.0)*sqrt2).astype(datatype)
            
            S1 = sigma[iL, jL, lL, mwing]
            S2 = sigma[iL, jL, mwing, lL]
            
          
            C3 = pref * (S1 + S2)
            C4 = pref4 * (S1 - S2)
            
            
            
            return A,B,C,D,C3,C4
        
        
        
        for tchunkL in nchunks:
            for tchunkR in nchunks:
                
                (indexL0, indexL1, chunkL) = tchunkL
                (indexR0, indexR1, chunkR) = tchunkR
                
                # indexL0=indexL
                # indexL1=indexL+chunk_size
                # indexR0=indexR
                # indexR1=indexR+chunk_size
                
                A,B,C,D,C3,C4=speedUpCore(chunkL,chunkR)
                
                r0 = 2 * indexL0
                r1 = 2 * indexL1
                c0 = 2 * indexR0
                c1 = 2 * indexR1
                
                # even-even (A)
                body[r0:r1:2, c0:c1:2] = A
                
                # even-odd (B)
                body[r0:r1:2, c0+1:c1:2] = B
                
                # odd-even (C)
                body[r0+1:r1:2, c0:c1:2] = C
                
                # odd-odd (D)
                body[r0+1:r1:2, c0+1:c1:2] = D
                
                wing[r0:r1:2, :] = C3
                wing[r0+1:r1:2, :] = C4     
 
        body[np.abs(body) < self.sparse_tol] = 0.0
        head[np.abs(head) < self.sparse_tol] = 0.0
        wing[np.abs(wing) < self.sparse_tol] = 0.0
        # body_sparse = csr_matrix(body)
        # print(wing.shape)
        # print(body.shape)
        # print(head.shape)
        
        
        H3upd=scipy.sparse.bmat([[head,np.transpose(wing)],[wing,body]], format='csr', dtype=datatype)
        
        # check16(H3upd)
        
        H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3upd,spaceObj,self.remove_single_values)
        
        self.timed("Creating spin Opt eff. Hamiltonian sparse",2)
        
        
        
        return H3upd
    
    
    
    def spinAdaptedMCDESparseOrig(self,secondBorn=False):
        """
        Construct the spin-adapted sparse MCDE effective Hamiltonian.
        
        This method transforms the spin-orbital three-particle basis into a
        spin-adapted spatial-orbital basis and constructs the corresponding
        MCDE effective Hamiltonian in sparse matrix format. The resulting matrix
        contains the one-particle sector, the spin-adapted three-particle sector,
        and the coupling between them.
        
        The spin-adapted basis is generated by collapsing spin-orbital indices
        onto spatial-orbital indices and removing duplicate configurations.
        For each spatial configuration, singlet-coupled and triplet-coupled
        three-particle states are constructed, yielding two spin-adapted states
        per spatial basis function.
        
        The effective Hamiltonian is assembled in block form,
        
        \[
        H_{\mathrm{MCDE}}
        =
        \\begin{pmatrix}
        H_{1p} & V^\dagger \\\\
        V & H_{3p}
        \\end{pmatrix}
        \]
        
        where $H_{1p}$ is the one-particle block, $H_{3p}$ is the
        spin-adapted three-particle block, and $V$ contains the coupling
        between the one- and three-particle sectors.
        
        Matrix elements smaller than `self.sparse_tol` are omitted during
        construction. After assembly, elements below `self.zero_tol` are
        removed and isolated diagonal states may optionally be eliminated using
        `self.AuxillaryFunctions.remove_isolated_diagonals_sparse`.
        
        Parameters:
            secondBorn : 
                Flag indicating whether the Second-Born approximation is used.
                This parameter is currently only employed for logging and consistency
                with other MCDE construction routines. Default is `False`. (bool, optional)
        
        Returns:
            matrix:
                Spin-adapted MCDE effective Hamiltonian in sparse CSR format. (scipy.sparse.csr_matrix)
        
        Notes:
            The size of the resulting Hamiltonian is
            
            $$
            N_{\mathrm{eff}}
            =
            N_{\mathrm{1p}}
            + 2 N_{\mathrm{3p}},
            $$
            
            where $N_{\mathrm{1p}}$ is the number of one-particle basis
            functions and $N_{\mathrm{3p}}$ is the number of unique spatial
            three-particle configurations.
            
            The factor of two arises from the two spin-adapted coupling channels
            associated with each spatial three-particle configuration.
            
            The method updates the internal attribute
            `self.nspace_spatials_MCDE` to reflect the reduced spin-adapted basis
            after any pruning operations.
        
        See Also:
            -spinAdaptedMCDE :
            Dense spin-adapted MCDE Hamiltonian construction.
            
            -spinAdaptedMCDESparse :
            Very fast, but memory expensive dense spin-adapted MCDE implementation.
            
            -spinAdaptedMCDESparseChunks :
            A speed and memory balanced implementation of the dense spin-adapted MCDE implementation.
            
            -spinAdaptedMCDESparseParallel :
            Parallelized verson of dense spin-adapted MCDE implementation.
            
            -spinAdaptedMCDEwithWSparse :
            like spinAdaptedMCDESparseOrig, but with the exchange V switched with W.
        """

        
        self.verbose3("Spin transformed MCDE sparse")
        self.verbose3("Second Born? "+str(secondBorn))
        
        datatype=self.data_type_sparse
        
        def d(a,b):
            return 1 if a==b else 0
        
        
        #transform spinorbitals to spatial orbitals
        nspace_spatials0=[]
        for entry in self.nspace:
            nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
        #remove duplicates
        seen = set()
        nspace_spatials = []
        for item in nspace_spatials0:
            t = tuple(item)
            if t not in seen:
                seen.add(t)
                nspace_spatials.append(item)
        
        spaceObj=MCDE.Nspace(np.arange(self.nBas),np.repeat(nspace_spatials,2,axis=0))
        
        self.verbose3("Length of 3-particle basis: " + str(len(nspace_spatials)))
        self.verbose4(nspace_spatials)
        ray=[]
        for i in range(self.nBas):
            ray.append(self.mo_en[i])
        head=np.diag(ray)
        
        #create wing
        
        # wing=np.zeros((len(nspace_spatials)*2,len(ray)))
        wing=lil_matrix((len(nspace_spatials)*2,len(ray)),dtype=datatype)
        
        for leftindex,left in enumerate(nspace_spatials):
            [i,j,l]=left
            for rightindex,m in enumerate(np.arange(self.nBas)):
                C3=np.sqrt(.5)**(d(i,j))*np.sqrt(.5)*(self.sigma_mo_gabi(i,j,l,m)+self.sigma_mo_gabi(i,j,m,l))
                C4=np.sqrt(3/2)*(self.sigma_mo_gabi(i,j,l,m)-self.sigma_mo_gabi(i,j,m,l))
                
                wing[2*leftindex,rightindex]=C3
                wing[2*leftindex+1,rightindex]=C4
        
        #create body matrix
        
        if self.mcde:
            # body=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            body=lil_matrix((len(nspace_spatials)*2,2*len(nspace_spatials)),dtype=datatype)
            # bodyHF=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            for leftindex,left in enumerate(nspace_spatials):
                [i,j,l]=left
                
                for rightindex,right in enumerate(nspace_spatials):
                    [m,o,k]=right
                    
                    
                    A1=np.sqrt(.5)**(d(m,o))*np.sqrt(.5)**(d(i,j))*(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)+self.sigma_mo_gabi(i,j,m,o))
                                                                  +d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-.5*self.sigma_mo_gabi(i,k,o,l))
                                                                  +d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-.5*self.sigma_mo_gabi(j,k,m,l))
                                                                  +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-.5*self.sigma_mo_gabi(i,k,m,l))
                                                                  +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-.5*self.sigma_mo_gabi(j,k,o,l)))
                    A2=(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)-self.sigma_mo_gabi(i,j,m,o))
                                                                  -d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-1.5*self.sigma_mo_gabi(i,k,o,l))
                                                                  -d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-1.5*self.sigma_mo_gabi(j,k,m,l))
                                                                  +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-1.5*self.sigma_mo_gabi(i,k,m,l))
                                                                  +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-1.5*self.sigma_mo_gabi(j,k,o,l)))
                    F1=np.sqrt(.5)**(d(i,j))*(np.sqrt(3)/2)*(-d(m,j)*self.sigma_mo_gabi(i,k,o,l)+d(i,o)*self.sigma_mo_gabi(j,k,m,l)
                                                          +d(o,j)*self.sigma_mo_gabi(i,k,m,l)
                                                          -d(i,m)*self.sigma_mo_gabi(j,k,o,l))
                    F2=np.sqrt(.5)**(d(m,o))*(np.sqrt(3)/2)*(d(m,j)*self.sigma_mo_gabi(i,k,o,l)-d(i,o)*self.sigma_mo_gabi(j,k,m,l)
                                                          +d(o,j)*self.sigma_mo_gabi(i,k,m,l)
                                                          -d(i,m)*self.sigma_mo_gabi(j,k,o,l))
                    
                    
                    fi = 0 if (i >= self.nO) else 1
                    fj = 0 if (j >= self.nO) else 1
                    fl = 0 if (l >= self.nO) else 1
                    prefac=-((1-fi)*(1-fj)*fl-fi*fj*(1-fl))
                    
                    ei=self.mo_en[i]+self.virtualShift(i)
                    ej=self.mo_en[j]+self.virtualShift(j)
                    el=self.mo_en[l]+self.virtualShift(l)
                    de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                    
                    A=de+prefac*A1
                    B=prefac*F1
                    C=prefac*F2
                    D=de+prefac*A2
                    
                    if abs(A)>self.sparse_tol:
                        body[2*leftindex,2*rightindex]=A
                    if abs(B)>self.sparse_tol:
                        body[2*leftindex,2*rightindex+1]=B
                    if abs(C)>self.sparse_tol:
                        body[2*leftindex+1,2*rightindex]=C
                    if abs(D)>self.sparse_tol:
                        body[2*leftindex+1,2*rightindex+1]=D
                    
                    # body[2*leftindex,2*rightindex]=de+prefac*A1
                    # body[2*leftindex,2*rightindex+1]=prefac*F1
                    # body[2*leftindex+1,2*rightindex]=prefac*F2
                    # body[2*leftindex+1,2*rightindex+1]=de+prefac*A2
                    
                    # bodyHF[2*leftindex,2*rightindex]=de
                    # bodyHF[2*leftindex+1,2*rightindex+1]=de
                    # H3upd=np.block([[head,np.transpose(wing)],[wing,body]])
            H3upd = scipy.sparse.bmat([[head, wing.T],
                  [wing, body]], format='csr',dtype=datatype)
            # H3upd[H3upd.abs() < self.zero_tol] = 0
            # H3upd.eliminate_zeros()
            mask = np.abs(H3upd.data) < self.zero_tol
            H3upd.data[mask] = 0
            # H3upd=np.where(np.abs(H3upd) < self.zero_tol, 0.0, H3upd)
            H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3upd,spaceObj,self.remove_single_values)
            
        if secondBorn:
            # bodySB=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            bodySB=lil_matrix((len(nspace_spatials)*2,2*len(nspace_spatials)),dtype=datatype)
            for leftindex,left in enumerate(nspace_spatials):
                [i,j,l]=left
                
                for rightindex,right in enumerate(nspace_spatials):
                    [m,o,k]=right
                    
                    
                    ei=self.mo_en[i]+self.virtualShift(i)
                    ej=self.mo_en[j]+self.virtualShift(j)
                    el=self.mo_en[l]+self.virtualShift(l)
                    de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                    
                    
                    bodySB[2*leftindex,2*rightindex]=de
                    bodySB[2*leftindex,2*rightindex+1]=0
                    bodySB[2*leftindex+1,2*rightindex]=0
                    bodySB[2*leftindex+1,2*rightindex+1]=de
            # H3SB=np.block([[head,np.transpose(wing)],[wing,bodySB]])
            H3SB=scipy.sparse.bmat([[head,np.transpose(wing)],[wing,bodySB]], format='csr', dtype=datatype)
            mask = np.abs(H3SB.data) < self.zero_tol
            H3SB.data[mask] = 0
            # H3SB=np.where(np.abs(H3SB) < self.zero_tol, 0.0, H3SB)
            # H3SB=H3SB.tocsr()
            H3SB,self.nspace_spatials_SB=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3SB,spaceObj,self.remove_single_values)

        
        
       
       
        
        
        self.timed("Creating spin Opt eff. Hamiltonian sparse",2)
        
        if secondBorn and self.mcde:
            return H3upd,H3SB
        if secondBorn:
            return H3SB
        if self.mcde:
            return H3upd

#%% spin adapted effective Hamiltonian with W replacing V_exchange
    def spinAdaptedMCDEwithWSparse(self,secondBorn=False):
        """
        Construct the spin-adapted sparse MCDE effective Hamiltonian with the 
        direct-$eh$ and direct- and exchange-$pp$ two electron integrals $V$ replaced
        by $W$ defined in `self.eri_mo_gabi_W`.
        
        The method works analogously to `self.spinAdaptedMCDESparseOrig`.
        
        Parameters:
            secondBorn : 
                Flag indicating whether the Second-Born approximation is used.
                This parameter is currently only employed for logging and consistency
                with other MCDE construction routines. Default is `False`. (bool, optional)
        
        Returns:
            matrix:
                Spin-adapted MCDE effective Hamiltonian in sparse CSR format. (scipy.sparse.csr_matrix)
        
        
        
        See Also:
            -spinAdaptedMCDE :
            Dense spin-adapted MCDE Hamiltonian construction.
            
            -spinAdaptedMCDESparse :
            Very fast, but memory expensive dense spin-adapted MCDE implementation.
            
            -spinAdaptedMCDESparseChunks :
            A speed and memory balanced implementation of the dense spin-adapted MCDE implementation.
            
            -spinAdaptedMCDESparseOrig :
            Slow sparse spin-adapted MCDE Hamiltonian construction.
            
            -spinAdaptedMCDESparseParallel :
            Parallelized verson of dense spin-adapted MCDE implementation.
        """
        # is W defined?
        if self.eri_mo_gabi_W is None:
            self.verbose1("No W defined, proceed with unscreened V.")
            return self.spinAdaptedMCDESparse(secondBorn)
        
        self.verbose3("Spin transformed MCDE sparse")
        self.verbose3("Second Born? "+str(secondBorn))
        
        datatype=self.data_type_sparse
        
        def d(a,b):
            return 1 if a==b else 0
        
        
        #transform spinorbitals to spatial orbitals
        nspace_spatials0=[]
        for entry in self.nspace:
            nspace_spatials0.append([entry[0]//2,entry[1]//2,entry[2]//2])
        #remove duplicates
        seen = set()
        nspace_spatials = []
        for item in nspace_spatials0:
            t = tuple(item)
            if t not in seen:
                seen.add(t)
                nspace_spatials.append(item)
        
        spaceObj=MCDE.Nspace(np.arange(self.nBas),np.repeat(nspace_spatials,2,axis=0))
        
        self.verbose3("Length of 3-particle basis: " + str(len(nspace_spatials)))
        self.verbose4(nspace_spatials)
        ray=[]
        for i in range(self.nBas):
            ray.append(self.mo_en[i])
        head=np.diag(ray)
        
        #create wing
        
        # wing=np.zeros((len(nspace_spatials)*2,len(ray)))
        wing=lil_matrix((len(nspace_spatials)*2,len(ray)),dtype=datatype)
        
        for leftindex,left in enumerate(nspace_spatials):
            [i,j,l]=left
            for rightindex,m in enumerate(np.arange(self.nBas)):
                C3=np.sqrt(.5)**(d(i,j))*np.sqrt(.5)*(self.sigma_mo_gabi(i,j,l,m)+self.sigma_mo_gabi_W(i,j,m,l))
                C4=np.sqrt(3/2)*(self.sigma_mo_gabi(i,j,l,m)-self.sigma_mo_gabi_W(i,j,m,l))
                
                wing[2*leftindex,rightindex]=C3
                wing[2*leftindex+1,rightindex]=C4
        
        #create body matrix
        
        if self.mcde:
            # body=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            body=lil_matrix((len(nspace_spatials)*2,2*len(nspace_spatials)),dtype=datatype)
            # bodyHF=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            for leftindex,left in enumerate(nspace_spatials):
                [i,j,l]=left
                
                for rightindex,right in enumerate(nspace_spatials):
                    [m,o,k]=right
                    
                    
                    A1=np.sqrt(.5)**(d(m,o))*np.sqrt(.5)**(d(i,j))*(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)+self.sigma_mo_gabi_W(i,j,m,o))
                                                                  +d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-.5*self.sigma_mo_gabi_W(i,k,o,l))
                                                                  +d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-.5*self.sigma_mo_gabi_W(j,k,m,l))
                                                                  +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-.5*self.sigma_mo_gabi_W(i,k,m,l))
                                                                  +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-.5*self.sigma_mo_gabi_W(j,k,o,l)))
                    A2=(-d(l,k)*(self.sigma_mo_gabi(i,j,o,m)-self.sigma_mo_gabi_W(i,j,m,o))
                                                                  -d(m,j)*(self.sigma_mo_gabi(i,k,l,o)-1.5*self.sigma_mo_gabi_W(i,k,o,l))
                                                                  -d(i,o)*(self.sigma_mo_gabi(j,k,l,m)-1.5*self.sigma_mo_gabi_W(j,k,m,l))
                                                                  +d(o,j)*(self.sigma_mo_gabi(i,k,l,m)-1.5*self.sigma_mo_gabi_W(i,k,m,l))
                                                                  +d(i,m)*(self.sigma_mo_gabi(j,k,l,o)-1.5*self.sigma_mo_gabi_W(j,k,o,l)))
                    F1=np.sqrt(.5)**(d(i,j))*(np.sqrt(3)/2)*(-d(m,j)*self.sigma_mo_gabi_W(i,k,o,l)+d(i,o)*self.sigma_mo_gabi_W(j,k,m,l)
                                                          +d(o,j)*self.sigma_mo_gabi_W(i,k,m,l)
                                                          -d(i,m)*self.sigma_mo_gabi_W(j,k,o,l))
                    F2=np.sqrt(.5)**(d(m,o))*(np.sqrt(3)/2)*(d(m,j)*self.sigma_mo_gabi_W(i,k,o,l)-d(i,o)*self.sigma_mo_gabi_W(j,k,m,l)
                                                          +d(o,j)*self.sigma_mo_gabi_W(i,k,m,l)
                                                          -d(i,m)*self.sigma_mo_gabi_W(j,k,o,l))
                    
                    
                    fi = 0 if (i >= self.nO) else 1
                    fj = 0 if (j >= self.nO) else 1
                    fl = 0 if (l >= self.nO) else 1
                    prefac=-((1-fi)*(1-fj)*fl-fi*fj*(1-fl))
                    
                    ei=self.mo_en[i]+self.virtualShift(i)
                    ej=self.mo_en[j]+self.virtualShift(j)
                    el=self.mo_en[l]+self.virtualShift(l)
                    de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                    
                    A=de+prefac*A1
                    B=prefac*F1
                    C=prefac*F2
                    D=de+prefac*A2
                    
                    if abs(A)>self.sparse_tol:
                        body[2*leftindex,2*rightindex]=A
                    if abs(B)>self.sparse_tol:
                        body[2*leftindex,2*rightindex+1]=B
                    if abs(C)>self.sparse_tol:
                        body[2*leftindex+1,2*rightindex]=C
                    if abs(D)>self.sparse_tol:
                        body[2*leftindex+1,2*rightindex+1]=D
                    
                    # body[2*leftindex,2*rightindex]=de+prefac*A1
                    # body[2*leftindex,2*rightindex+1]=prefac*F1
                    # body[2*leftindex+1,2*rightindex]=prefac*F2
                    # body[2*leftindex+1,2*rightindex+1]=de+prefac*A2
                    
                    # bodyHF[2*leftindex,2*rightindex]=de
                    # bodyHF[2*leftindex+1,2*rightindex+1]=de
                    # H3upd=np.block([[head,np.transpose(wing)],[wing,body]])
            H3upd = scipy.sparse.bmat([[head, wing.T],
                  [wing, body]], format='csr',dtype=datatype)
            # H3upd[H3upd.abs() < self.zero_tol] = 0
            # H3upd.eliminate_zeros()
            mask = np.abs(H3upd.data) < self.zero_tol
            H3upd.data[mask] = 0
            # H3upd=np.where(np.abs(H3upd) < self.zero_tol, 0.0, H3upd)
            H3upd,self.nspace_spatials_MCDE=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3upd,spaceObj,self.remove_single_values)
            
        if secondBorn:
            # bodySB=np.zeros((len(nspace_spatials)*2,2*len(nspace_spatials)))
            bodySB=lil_matrix((len(nspace_spatials)*2,2*len(nspace_spatials)),dtype=datatype)
            for leftindex,left in enumerate(nspace_spatials):
                [i,j,l]=left
                
                for rightindex,right in enumerate(nspace_spatials):
                    [m,o,k]=right
                    
                    
                    ei=self.mo_en[i]+self.virtualShift(i)
                    ej=self.mo_en[j]+self.virtualShift(j)
                    el=self.mo_en[l]+self.virtualShift(l)
                    de=(ei-(el-ej))*d(i,m)*d(j,o)*d(l,k)
                    
                    
                    bodySB[2*leftindex,2*rightindex]=de
                    bodySB[2*leftindex,2*rightindex+1]=0
                    bodySB[2*leftindex+1,2*rightindex]=0
                    bodySB[2*leftindex+1,2*rightindex+1]=de
            # H3SB=np.block([[head,np.transpose(wing)],[wing,bodySB]])
            H3SB=scipy.sparse.bmat([[head,np.transpose(wing)],[wing,bodySB]], format='csr', dtype=datatype)
            mask = np.abs(H3SB.data) < self.zero_tol
            H3SB.data[mask] = 0
            # H3SB=np.where(np.abs(H3SB) < self.zero_tol, 0.0, H3SB)
            # H3SB=H3SB.tocsr()
            H3SB,self.nspace_spatials_SB=self.AuxillaryFunctions.remove_isolated_diagonals_sparse(H3SB,spaceObj,self.remove_single_values)
    
        
        
       
       
        
        
        self.timed("Creating spin Opt eff. Hamiltonian sparse",2)
        
        if secondBorn and self.mcde:
            return H3upd,H3SB
        if secondBorn:
            return H3SB
        if self.mcde:
            return H3upd

#%% How to diagonalize the effective Hamiltonian
    
    def exactDiagonalization(self,matrix):
        """
        Compute the exact diagonalization of a `numpy.array` effective Hamiltonian `matrix`.
        Returns eigenvalues and eigenvectors.
 
        Parameters:
            matrix : effective Hamiltonian as a numpy.array (shape: ($\lambda$,$\lambda$))
            

        Returns:
            evals       : sorted eigenenergies of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$))
            evecs       : sorted column-eigenvectors of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$,$\lambda$))
        """
        evals,evecs=self.AuxillaryFunctions.eig(matrix)

        self.verbose4("Eigenvalues for H effective")
        if self.verbose>=4:
            idx = np.argsort(evals)  # use -eigvals for descending
            eigvals_sorted = evals[idx]
            # eigvecs_sorted = evecs[:, idx]
            for ii in range(len(evals)):
                self.verbose4('%4d      %.5f'%(ii,eigvals_sorted[ii]))
        
        return evals,evecs
    
    def exactDiagonalizationSparse(self,matrix):
        """
        Compute the exact diagonalization of a `scipy.sparse` effective Hamiltonian `matrix`.
        Returns eigenvalues and eigenvectors.
        If `self.reduce_evecs_to_1body`, the three-particle part of the eigenvectors is discarded. It only impacts the memory usage.
 
        Parameters:
            matrix : effective Hamiltonian as a scipy.sparse csr object (shape: ($\lambda$,$\lambda$))
            

        Returns:
            evals       : sorted eigenenergies of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$))
            evecs       : sorted column-eigenvectors of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$,$\lambda$))
        """
        vecstocalc=min(self.do_auto_sparse_basis_threshold,self.estimate_spin_opt_ham)
        self.verbose3(f"Calculating {vecstocalc} eigenvectors")
        if vecstocalc>=self.estimate_spin_opt_ham:
            evals,evecs=self.AuxillaryFunctions.eig(matrix.toarray())
        else:
            evals,evecs=scipy.sparse.linalg.eigsh(matrix,k=vecstocalc, which='LM')
        
        if self.reduce_evecs_to_1body:
            evecs=evecs[:self.nBas]
        self.verbose4("Eigenvalues for H effective")
        if self.verbose>=4:
            idx = np.argsort(evals)  # use -eigvals for descending
            eigvals_sorted = evals[idx]
            # eigvecs_sorted = evecs[:, idx]
            for ii in range(len(evals)):
                self.verbose4('%4d      %.5f'%(ii,eigvals_sorted[ii]))
        self.timed("Calculation of eigenvectors", 3)
        return evals,evecs
    
    def LanczosAlgorithm(self,matrix,nbas):
        """
        Compute the Haydock-Lanczos algorithm for an effective Hamiltonian `matrix` up to `self.iterations`.
        The Lanczos algorithm starts from an initial three-particle guess vector $\psi_0$, which has 1 for the one-particle coefficients (up to `nbas`), and 
        0 for the three-particle coefficients. $\psi_0$ is multiplied with the effective Hamiltonian $H$ (`matrix`) to generate an approximate vector $\psi_1$.
        From $\psi_0$ and $\psi_1$ Lanczos coefficients $a,b$ are won. Then, the algorithm is repeated iteratively,
        with $\psi_{i+1}=H\psi_{i}$, until either `self.iterations` iterations are done, or the norm of $\psi_{i+1}$ is smaller
        than `self.lanczos_tol`, in which case an invariant subspace of $H$ has been probed.
        For each iteration step, a full Gram-Schmidt reorthogonalization of the Lanczos vectors $\psi_0,...,\psi_{i+1}$ is performed
 
        Parameters:
            matrix : effective Hamiltonian as a numpy array (shape: (M,M))
            nbas       : number of one-particle basis states. For the full effective Hamiltonian it is twice the number of restricted Hartree-Fock spatial orbitals. For the spin-adapted
            Hamiltonian it is the number of restricted Hartree-Fock spatial orbitals. (int)

        Returns:
            a       : array of diagonal terms [a1, a2, ..., an] of the Lanczos tridiagonal matrix (length n)
            b       : array of off-diagonal terms [b1, b2, ..., bn] of the Lanczos tridiagonal matrix. (length n)
        """
        self.verbose1("Starting Lanczos Algorithm")
        
        acoeff = np.zeros(self.iterations)
        bcoeff = np.zeros(self.iterations)  # will ignore bcoeff[0]
        lanczosBasis = np.zeros((self.iterations, len(matrix)))
        
        # Start vector
        s0 = np.zeros(len(matrix))
        s0[:nbas] = 1
        s0 /= np.linalg.norm(s0)
        lanczosBasis[0] = s0
        
        # First step
        w = matrix @ s0
        acoeff[0] = np.dot(s0, w)
        w -= acoeff[0] * s0
        bcoeff[1] = np.linalg.norm(w)
        lanczosBasis[1] = w / bcoeff[1]
        
        # Main loop
        for i in range(1, self.iterations - 1):
            v_prev = lanczosBasis[i - 1]
            v_curr = lanczosBasis[i]
            w = matrix @ v_curr
            acoeff[i] = np.dot(v_curr, w)
            w -= acoeff[i] * v_curr + bcoeff[i] * v_prev
        
            # Optional: reorthogonalize w to all previous basis vectors
            # for j in range(i):
            #     w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            # FULL REORTHOGONALIZATION
            for j in range(i + 1):
                w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            bcoeff[i + 1] = np.linalg.norm(w)
            if bcoeff[i + 1] < self.lanczos_tol:
                self.verbose2("Breakdown at step "+ str(i))
                break
            lanczosBasis[i + 1] = w / bcoeff[i + 1]
        
        acoeff[self.iterations - 1] = lanczosBasis[self.iterations - 1] @ matrix @ lanczosBasis[self.iterations - 1].T
        
        self.timed("Lanczos Algorithm", 2)
        
        return acoeff,bcoeff
    
    def LanczosAlgorithmSparse0(self,matrix,nbas):
        """
        Compute the Haydock-Lanczos algorithm for a sparse effective Hamiltonian `matrix` up to `self.iterations`.
        The Lanczos algorithm starts from an initial three-particle guess vector $\psi_0$, which has 1 for the one-particle coefficients (up to `nbas`), and 
        0 for the three-particle coefficients. $\psi_0$ is multiplied with the effective Hamiltonian $H$ (`matrix`) to generate an approximate vector $\psi_1$.
        From $\psi_0$ and $\psi_1$ Lanczos coefficients $a,b$ are won. Then, the algorithm is repeated iteratively,
        with $\psi_{i+1}=H\psi_{i}$, until either `self.iterations` iterations are done, or the norm of $\psi_{i+1}$ is smaller
        than `self.lanczos_tol`, in which case an invariant subspace of $H$ has been probed.
        For each iteration step, a full Gram-Schmidt reorthogonalization of the Lanczos vectors $\psi_0,...,\psi_{i+1}$ is performed
 
        Parameters:
            matrix : effective Hamiltonian as a `scipy.sparse` csr matrix (shape: (M,M))
            nbas       : number of one-particle basis states. For the full effective Hamiltonian it is twice the number of restricted Hartree-Fock spatial orbitals. For the spin-adapted
            Hamiltonian it is the number of restricted Hartree-Fock spatial orbitals. (int)

        Returns:
            a       : array of diagonal terms [a1, a2, ..., an] of the Lanczos tridiagonal matrix (length n)
            b       : array of off-diagonal terms [b1, b2, ..., bn] of the Lanczos tridiagonal matrix. (length n)
        """
        self.verbose1("Starting Lanczos Algorithm")
        
        matrix_length=matrix.shape[0]
        
        acoeff = np.zeros(self.iterations)
        bcoeff = np.zeros(self.iterations)  # will ignore bcoeff[0]
        lanczosBasis = np.zeros((self.iterations, matrix_length))
        
        # Start vector
        s0 = np.zeros(matrix_length)
        s0[:nbas] = 1
        s0 /= np.linalg.norm(s0)
        lanczosBasis[0] = s0
        
        # First step
        w = matrix @ s0
        acoeff[0] = np.dot(s0, w)
        w -= acoeff[0] * s0
        bcoeff[1] = np.linalg.norm(w)
        lanczosBasis[1] = w / bcoeff[1]
        
        # Main loop
        for i in range(1, self.iterations - 1):
            v_prev = lanczosBasis[i - 1]
            v_curr = lanczosBasis[i]
            w = matrix @ v_curr
            acoeff[i] = np.dot(v_curr, w)
            w -= acoeff[i] * v_curr + bcoeff[i] * v_prev
        
            # Optional: reorthogonalize w to all previous basis vectors
            # for j in range(i):
            #     w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            # FULL REORTHOGONALIZATION
            for j in range(i + 1):
                w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            bcoeff[i + 1] = np.linalg.norm(w)
            if bcoeff[i + 1] < self.lanczos_tol:
                self.verbose2("Breakdown at step "+ str(i))
                break
            lanczosBasis[i + 1] = w / bcoeff[i + 1]
        
        self.w=lanczosBasis
        
        acoeff[self.iterations - 1] = lanczosBasis[self.iterations - 1] @ matrix @ lanczosBasis[self.iterations - 1].T
        
        self.timed("Lanczos Algorithm", 2)
        
        return acoeff,bcoeff
    
    
    
    def LanczosAlgorithmSparse(self,matrix,nbas,returnLanczosVectors=False):
        """
        Compute the Haydock-Lanczos algorithm for a sparse effective Hamiltonian `matrix` up to `self.iterations`.
        The Lanczos algorithm starts from an initial three-particle guess vector $\psi_0$, which has 1 for the one-particle coefficients (up to `nbas`), and 
        0 for the three-particle coefficients. $\psi_0$ is multiplied with the effective Hamiltonian $H$ (`matrix`) to generate an approximate vector $\psi_1$.
        From $\psi_0$ and $\psi_1$ Lanczos coefficients $a,b$ are won. Then, the algorithm is repeated iteratively,
        with $\psi_{i+1}=H\psi_{i}$, until either `self.iterations` iterations are done, or the norm of $\psi_{i+1}$ is smaller
        than `self.lanczos_tol`, in which case an invariant subspace of $H$ has been probed.
        For each iteration step, a full Gram-Schmidt reorthogonalization of the Lanczos vectors $\psi_0,...,\psi_{i+1}$ is performed
 
        Parameters:
            matrix : effective Hamiltonian as a `scipy.sparse` csr matrix (shape: (M,M))
            nbas       : number of one-particle basis states. For the full effective Hamiltonian it is twice the number of restricted Hartree-Fock spatial orbitals. For the spin-adapted
            Hamiltonian it is the number of restricted Hartree-Fock spatial orbitals. (int)
            returnLanczosVectors : True if you want to return the Lanczos vectors of the Krylov Basis. (Bool)

        Returns:
            a       : array of diagonal terms [a1, a2, ..., an] of the Lanczos tridiagonal matrix (length n)
            b       : array of off-diagonal terms [b1, b2, ..., bn] of the Lanczos tridiagonal matrix. (length n)
            lv (optional) : if returnLanczosVectors is True, returns an array of row Lanczos vectors (length (n,M))
        """
        self.verbose1("Starting Lanczos Algorithm")
        
        matrix_length=matrix.shape[0]
        
        acoeff = np.zeros(self.iterations)
        bcoeff = np.zeros(self.iterations)  # will ignore bcoeff[0]
        lanczosBasis = np.zeros((self.iterations, matrix_length))
        
        # Start vector
        s0 = np.zeros(matrix_length)
        s0[:nbas] = 1
        s0 /= np.linalg.norm(s0)
        lanczosBasis[0] = s0
        
        # First step
        w = matrix @ s0
        acoeff[0] = np.dot(s0, w)
        w -= acoeff[0] * s0
        bcoeff[1] = np.linalg.norm(w)
        lanczosBasis[1] = w / bcoeff[1]
        
        # Main loop
        final_index = self.iterations - 1
        for i in range(1, self.iterations - 1):
            v_prev = lanczosBasis[i - 1]
            v_curr = lanczosBasis[i]
            w = matrix @ v_curr
            acoeff[i] = np.dot(v_curr, w)
            w -= acoeff[i] * v_curr + bcoeff[i] * v_prev
        
            # Optional: reorthogonalize w to all previous basis vectors
            # for j in range(i):
            #     w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            # FULL REORTHOGONALIZATION
            for j in range(i + 1):
                w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            b = np.linalg.norm(w)
            # bcoeff[i + 1] = np.linalg.norm(w)
            if b < self.lanczos_tol:
                self.verbose2("Breakdown at step "+ str(i))
                final_index=i
                break
            lanczosBasis[i + 1] = w / b
            bcoeff[i+1]=b
            final_index = i+1
        self.w=lanczosBasis
        
        acoeff[final_index] = lanczosBasis[final_index] @ matrix @ lanczosBasis[final_index].T

        
        self.timed("Lanczos Algorithm", 2)
        
        if returnLanczosVectors:
            return acoeff[:final_index+1], bcoeff[:final_index+1], lanczosBasis 
        return acoeff[:final_index+1], bcoeff[:final_index+1]
    
    def LanczosAlgorithmSparseNoOrth(self,matrix,nbas,returnLanczosVectors=False):
        """
        Compute the Haydock-Lanczos algorithm for a sparse effective Hamiltonian up to `self.iterations`.
        No Gram-Schmidt reorthogonalization is applied.
        The Lanczos algorithm starts from an initial three-particle guess vector $\psi_0$, which has 1 for the one-particle coefficients (up to `nbas`), and 
        0 for the three-particle coefficients. $\psi_0$ is multiplied with the effective Hamiltonian $H$ (`matrix`) to generate an approximate vector $\psi_1$.
        From $\psi_0$ and $\psi_1$ Lanczos coefficients $a,b$ are won. Then, the algorithm is repeated iteratively,
        with $\psi_{i+1}=H\psi_{i}$, until either `self.iterations` iterations are done, or the norm of $\psi_{i+1}$ is smaller
        than `self.lanczos_tol`, in which case an invariant subspace of $H$ has been probed.
 
        Parameters:
            matrix : effective Hamiltonian as a `scipy.sparse` csr matrix (shape: (M,M))
            nbas       : number of one-particle basis states. For the full effective Hamiltonian it is twice the number of restricted Hartree-Fock spatial orbitals. For the spin-adapted
            Hamiltonian it is the number of restricted Hartree-Fock spatial orbitals. (int)
            returnLanczosVectors : True if you want to return the Lanczos vectors of the Krylov Basis. (Bool)

        Returns:
            a       : array of diagonal terms [a1, a2, ..., an] of the Lanczos tridiagonal matrix (length n)
            b       : array of off-diagonal terms [b1, b2, ..., bn] of the Lanczos tridiagonal matrix. (length n)
            lv (optional) : if returnLanczosVectors is True, returns an array of row Lanczos vectors (length (n,M))
            
        Notes:
            No orthogonalization is performed.
        """
        self.verbose1("Starting Lanczos Algorithm")
        
        matrix_length=matrix.shape[0]
        
        acoeff = np.zeros(self.iterations)
        bcoeff = np.zeros(self.iterations)  # will ignore bcoeff[0]
        lanczosBasis = np.zeros((self.iterations, matrix_length))
        
        # Start vector
        s0 = np.zeros(matrix_length)
        s0[:nbas] = 1
        s0 /= np.linalg.norm(s0)
        lanczosBasis[0] = s0
        
        # First step
        w = matrix @ s0
        acoeff[0] = np.dot(s0, w)
        w -= acoeff[0] * s0
        bcoeff[1] = np.linalg.norm(w)
        lanczosBasis[1] = w / bcoeff[1]
        
        # Main loop
        final_index = self.iterations - 1
        for i in range(1, self.iterations - 1):
            v_prev = lanczosBasis[i - 1]
            v_curr = lanczosBasis[i]
            w = matrix @ v_curr
            acoeff[i] = np.dot(v_curr, w)
            w -= acoeff[i] * v_curr + bcoeff[i] * v_prev
        
            # Optional: reorthogonalize w to all previous basis vectors
            # for j in range(i):
            #     w -= np.dot(lanczosBasis[j], w) * lanczosBasis[j]
        
            
        
            b = np.linalg.norm(w)
            # bcoeff[i + 1] = np.linalg.norm(w)
            if b < self.lanczos_tol:
                self.verbose2("Breakdown at step "+ str(i))
                final_index=i
                break
            lanczosBasis[i + 1] = w / b
            bcoeff[i+1]=b
            final_index = i+1
        self.w=lanczosBasis
        
        acoeff[final_index] = lanczosBasis[final_index] @ matrix @ lanczosBasis[final_index].T

        
        self.timed("Lanczos Algorithm", 2)
        
        if returnLanczosVectors:
            return acoeff[:final_index+1], bcoeff[:final_index+1], lanczosBasis 
        return acoeff[:final_index+1], bcoeff[:final_index+1]
#%% plotting
    @staticmethod
    def AA_vectorized(omega_array, eta, eig31, evec31, nBas):
        """
        Generates the spectrum $A(\omega)$ for the one-particle space.
        The spectrum is evaluated over the three-particle space according to
        $$
        A^{\text{1p}}(\omega)=\frac{1}{\pi}\sum_{i} |\im G^{\text{1p}}_{3,(i;i)}(\omega)|,
        $$
        with 
        
        $$
        G^{\text{1p}}_{3,(i;m)}(\omega)& =  \sum_{\lambda}\frac{A^{i}_{\lambda}A^{*m}_{\lambda}}{\omega-E_{\lambda}}.
        $$
        The $A^{i}_{\lambda},A^{*m}_{\lambda}$ are the one-particle part of the effective Hamiltonian's $\lambda$th eigenvectors. $\omega_{\lambda}$
        is the $\lambda$th eigenenergy.
 
        Parameters:
            omega_array : array of float energies over which the spectrum is calculated (shape: (M,))
            eta : Lorentzian broadening (float)
            eig31       : eigenenergies of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$))
            evec31       : row-eigenvectors of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$,$\lambda$))
            nBas       : number of one-particle basis states. For the full effective Hamiltonian it is twice the number of restricted Hartree-Fock spatial orbitals. For the spin-adapted
            Hamiltonian it is the number of restricted Hartree-Fock spatial orbitals. (int)

        Returns:
            A(omega_array) : array of real values representing the one-particle spectrum over the energy range omega_array  (shape: (M,))
        """
        print("Calculating the spectrum")
        start_time=time.time()
        evec_squared = evec31[:nBas, :] * evec31[:nBas, :].conj()  # shape (nBas, nEig)
        
        denom = eig31[:, np.newaxis].real  # shape (nEig, 1)
    
        omega_eta = omega_array[np.newaxis, :] - denom + 1j*eta # shape (nEig, N_omega)
    
        weights = np.sum(evec_squared, axis=0)  # shape (nEig,)
    
        
        response = np.sum(weights[:, np.newaxis] / omega_eta, axis=0)  # shape (N_omega,)
        
        amplitudes=-1 / np.pi * np.imag(response)
        result = np.column_stack((omega_array.flatten(), amplitudes))
        end_time=time.time()
        elapsed=end_time-start_time
        print(f"Calculating spectrum took {elapsed:.2f} seconds")
        return result
    
    @staticmethod
    def AA_Lanczos(omega_array,eta, a, b):
        """
        Generates the spectrum $A(\omega)$ for the one-particle space for Lanczos parameters.
        Evaluation of the continued fraction of the Lanczos tridiagonal matrix elements over an energy range omega_array.
        Returns a spectrum $A(z)$ as an array.
        $z$ represents $\omega +i\eta$, with $\eta$ the Lorentzian broadening, and $\omega$ the energies.
        The spectrum formula used is:
        $$
        A(z) = -\pi \, \mathrm{Im}[ z - a_0 - \cfrac{b_1^2}{z - a_1 - \cfrac{b_2^2}{z - a_2 - \cdots}} ]^{-1}
        $$
        
        Parameters:
            omega_array : array of float energies over which the spectrum is calculated (shape: (M,))
            eta : Lorentzian broadening (float)
            a       : array of diagonal terms [a1, a2, ..., an] of the Lanczos tridiagonal matrix (length n)
            b       : array of off-diagonal terms [b1, b2, ..., bn] of the Lanczos tridiagonal matrix. The first term b1 
            is automatically discarded (length n)

        Returns:
            A(omega_array) : array of real values representing the spectrum over the energy range omega_array  (shape: (M,))
        """
        b=b[1:]
        z_array=omega_array + 1j*eta
        z_array = np.asarray(z_array)
        result = z_array - a[-1]

        for i in reversed(range(len(b))):
            result = z_array - a[i] - b[i]**2 / result
        result = np.column_stack((z_array.real, -np.pi*(1/result).imag))
        return result
    
    @staticmethod
    def AA_vectorized_3p(omega_array, eta, eig31, evec31, nBas):
        """
        Generates the spectrum $A(\omega)$ for the three-particle space.
        The spectrum is evaluated over the three-particle space according to
        $$
        A^{\text{3p}}(\omega)=\frac{1}{\pi}\sum_{ijk} |\im G^{\text{3p}}_{3,(ijk;ijk)}(\omega)|,
        $$
        with 
        
        $$
        G^{\text{3p}}_{3,(ijl;mok)}(\omega)& =  \sum_{\lambda}\frac{A^{ijl}_{\lambda}A^{*mok}_{\lambda}}{\omega-E_{\lambda}}.
        $$
        The $A^{ijl}_{\lambda},A^{*mok}_{\lambda}$ are the three-particle part of the effective Hamiltonian's $\lambda$th eigenvectors. $\omega_{\lambda}$
        is the $\lambda$th eigenenergy.
 
        Parameters:
            omega_array : array of float energies over which the spectrum is calculated (shape: (M,))
            eta : Lorentzian broadening (float)
            eig31       : eigenenergies of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$))
            evec31       : row-eigenvectors of effective Hamiltonian. Their size depends on the effective Hamiltonian calculated, but they all have in common that 
            the first nBas entries pertain to the one-particle space (shape ($\lambda$,$\lambda$))
            nBas       : number of one-particle basis states. For the full effective Hamiltonian it is twice the number of restricted Hartree-Fock spatial orbitals. For the spin-adapted
            Hamiltonian it is the number of restricted Hartree-Fock spatial orbitals. (int)

        Returns:
            A(omega_array) : array of real values representing the three-particle spectrum over the energy range omega_array  (shape: (M,))
        """
        print("Calculating the spectrum - only 3p")
        start_time=time.time()
        evec_squared = evec31[nBas:, :] * evec31[nBas:, :].conj()  # shape (nBas, nEig)
        
        denom = eig31[:, np.newaxis].real  # shape (nEig, 1)
    
        # omega_eta = omega_array[np.newaxis, :] - denom + eta_arr[:, np.newaxis] # shape (nEig, N_omega)
        
        omega_eta = omega_array[np.newaxis, :] - denom + 1j*eta # shape (nEig, N_omega)
    
        weights = np.sum(evec_squared, axis=0)  # shape (nEig,)
    
        # # for i, (E, w) in enumerate(zip(eig31*Hartree, weights)):
        # #     print(f"Excitation {i}: E = {E:.3f}, weight = {w:.6f}")
        # for i, (E, w) in enumerate(zip(eig31-eta_arr, weights)):
        #     contrib = -1/np.pi * np.imag(w / (omega_array - E))
        #     plt.plot(omega_array, contrib, label=f"Exc {i}, w={w:.3e}")
        #     # plt.savefig("bse_vs_dyson/supp_"+mol_name+"_"+basis+"_"+nnn+".png", format='png', dpi=300)
        response = np.sum(weights[:, np.newaxis] / omega_eta, axis=0)  # shape (N_omega,)
        
        amplitudes=-1 / np.pi * np.imag(response)
        result = np.column_stack((omega_array.flatten(), amplitudes))
        end_time=time.time()
        elapsed=end_time-start_time
        print(f"Calculating spectrum took {elapsed:.2f} seconds")
        return result
    
#%% contribution analysis

    def getTopContributionsInEvec(self, eigen, evecs0, ao_labels=None, secondBorn=None,topN=3, MINVEC=-1, MAXVEC=-1,roundingLevel=5):
        """
        Determine the dominant basis-state contributions to effective Hamiltonian eigenvectors.
        
        For each eigenvector, the squared amplitudes :math:`|c_i|^2` of the basis-state
        coefficients are computed and the ``topN`` largest contributions are retained.
        Eigenvectors with degenerate (or nearly degenerate) eigenenergies are grouped
        according to their energy rounded to ``roundingLevel`` decimal places. The
        contributions of all eigenvectors within a group are summed and normalized.
        
        The resulting assignments can be returned either in terms of basis-state
        indices or translated to atomic-orbital labels if ``ao_labels`` is provided.
        
        Parameters
        ----------
        eigen : numpy.ndarray
            Eigenenergies of the effective Hamiltonian with shape ``(N,)``.
        
        evecs0 : numpy.ndarray
            Matrix of eigenvectors of the effective Hamiltonian. The expected shape is
            ``(N, N)`` with eigenvectors stored as columns. Internally, the matrix is
            transposed such that individual eigenvectors are processed row-wise.
        
        ao_labels : list[str] | None, optional
            Atomic-orbital labels used to translate basis-state indices into a
            human-readable representation. If ``None``, numerical labels are returned.
            Default is ``None``.
        
        secondBorn : bool | None, optional
            Whether the Second-Born basis-space mapping should be used when
            translating basis-state indices. If ``None``, the value of
            ``self.secondBorn`` is used. Default is ``None``.
        
        topN : int, optional
            Number of largest basis-state contributions retained for each eigenvector.
            Default is ``3``.
        
        MINVEC : int, optional
            Index of the first eigenvector to analyze. If negative, analysis starts
            from the first eigenvector. Default is ``-1``.
        
        MAXVEC : int, optional
            Index one past the last eigenvector to analyze. If negative, all
            eigenvectors are included. Default is ``-1``.
        
        roundingLevel : int, optional
            Number of decimal places used when grouping nearly degenerate
            eigenenergies. Default is ``5``.
        
        Returns
        -------
        dict[float, dict]
            Dictionary mapping rounded eigenenergies to normalized contribution
            dictionaries.
        
            The outer dictionary has the form
        
            .. code-block:: python
        
                {
                    energy_1: {label_1: weight_1, label_2: weight_2, ...},
                    energy_2: {label_1: weight_1, label_2: weight_2, ...},
                    ...
                }
        
            where the weights correspond to normalized summed contributions
            :math:`|c_i|^2` of the dominant basis states.
        
        Notes
        -----
        Only the ``topN`` largest contributions of each eigenvector are retained
        before grouping and normalization. Consequently, the returned weights
        represent the relative importance of the dominant basis states rather than
        the complete decomposition of the eigenvector.
        
        Degeneracies are identified by rounding eigenenergies to
        ``roundingLevel`` decimal places.
        """
       
        if secondBorn==None:
            secondBorn=self.secondBorn
            
        evecs0 = evecs0.T
        if ao_labels is not None:
            ao_labels = [lbl.rstrip() for lbl in ao_labels]
        
        if MINVEC < 0:
            MINVEC = 0
        if MAXVEC < 0:
            MAXVEC = len(evecs0)
    
        # Dictionary: rounded_energy → list of contribution dicts
        # Each dict: { index : magnitude }
        grouped = defaultdict(list)
    
        for i in range(MINVEC, MAXVEC):
    
            eigval = eigen[i]
            evec = evecs0[i]
    
            # 1. Take probabilities |c|² for the eigenvector
            probs = np.abs(evec) ** 2
    
            # 2. Put (prob, index) pairs and take the top N
            pairs = list(zip(probs, range(len(probs))))
            pairs.sort(reverse=True, key=lambda x: x[0])
            top = pairs[:topN]
    
            # 3. Build structure: [eigval, (prob1, idx1), ...]
            rounded_energy = round(float(eigval), roundingLevel)  # rounding level adjustable
    
            # 4. Store this eigenvector's contributions
            contrib_dict = {idx: prob for prob, idx in top}
            grouped[rounded_energy].append(contrib_dict)
    
        # 5. Sum + normalize contributions for each degenerate energy
        result = {}
    
        for energy, contrib_list in grouped.items():
            combined = defaultdict(float)
    
            # Sum contributions of all eigenvectors belonging to this energy
            for contrib in contrib_list:
                for idx, mag in contrib.items():
                    combined[idx] += mag
    
            # Normalize
            norm = sum(combined.values())
            if norm > 0:
                for k in combined:
                    combined[k] /= norm
    
            result[energy] = dict(combined)
        
        if ao_labels is not None:
            if not secondBorn:
                for energy in result:
                    result[energy] = {self.nspace_spatials_MCDE.translateToAOLabels(idx,ao_labels): val for idx, val in result[energy].items()}
            else:
                for energy in result:
                    result[energy] = {self.nspace_spatials_SB.translateToAOLabels(idx,ao_labels): val for idx, val in result[energy].items()}
        else:
            for energy in result:
                result[energy] = {self.nspace_spatials_MCDE.returnNumLabels(idx): val for idx, val in result[energy].items()}
        return result
        
    def getTopContributionsIn1Evec(self, eigen, evecs0, topN=10, ao_labels=None, secondBorn=None, roundingLevel=5):
        """
        Determine the dominant one-particle contributions to effective Hamiltonian eigenvectors.
        
        For each eigenvector, the squared amplitudes :math:`|c_i|^2` of the basis-state
        coefficients are computed and sorted in descending order. The `topN` largest
        contributions are retained for each eigenvector. Eigenvectors with degenerate
        (or nearly degenerate) eigenenergies are grouped according to their energy
        rounded to `roundingLevel` decimal places, and contributions from all
        eigenvectors in the group are summed.
        
        Unlike :meth:`getTopContributionsInEvec`, the summed contributions are not
        normalized. The results are returned as sorted lists of basis-state labels and
        their associated weights.
        
        Parameters
        -----
        eigen : numpy.ndarray
            Eigenenergies of the effective Hamiltonian with shape `(N,)`.
        
        evecs0 : numpy.ndarray
            Matrix of eigenvectors of the effective Hamiltonian. The expected shape is
            `(N, N)` with eigenvectors stored as columns. Internally, the matrix is
            transposed such that individual eigenvectors are processed row-wise.
        
        topN : int, optional
            Number of largest basis-state contributions retained for each eigenvector.
            If `topN <= 0`, all contributions are retained. Default is `10`.
        
        ao_labels : list[str] | None, optional
            Atomic-orbital labels. Currently unused by this method but retained for
            interface compatibility. Default is `None`.
        
        secondBorn : bool | None, optional
            If `None`, the value of `self.secondBorn` is used. Currently unused in
            the returned result but retained for interface compatibility.
            Default is `None`.
        
        roundingLevel : int, optional
            Number of decimal places used when grouping nearly degenerate
            eigenenergies. Default is `5`.
        
        Returns
        -----
        dict[float, list[tuple[str, float]]]
            Dictionary mapping rounded eigenenergies to sorted lists of
            `(label, weight)` pairs.
        
        ```
        The outer dictionary has the form
        
        .. code-block:: python
        
            {
                energy_1: [
                    ("label_1", weight_1),
                    ("label_2", weight_2),
                    ...
                ],
                energy_2: [
                    ("label_1", weight_1),
                    ("label_2", weight_2),
                    ...
                ],
                ...
            }
        
        where the labels correspond to basis-state occupations returned by
        ``self.nspace_spatials_MCDE.returnNumLabels`` and the weights are summed
        contributions :math:`|c_i|^2`.
        ```
        
        Notes
        -----
        Eigenvectors are grouped according to energies rounded to
        `roundingLevel` decimal places.
        
        The returned weights are not normalized. Consequently, the total weight
        associated with a given energy depends on both the number of grouped
        eigenvectors and the retained contributions.
        
        If `topN <= 0`, all basis-state contributions are retained before
        grouping.
        """

        if secondBorn==None:
            secondBorn=self.secondBorn
            
        evecs0 = evecs0.T
        if ao_labels is not None:
            ao_labels = [lbl.rstrip() for lbl in ao_labels]
        
        
       
        MINVEC = 0
        MAXVEC = len(evecs0)
   
        # Dictionary: rounded_energy → list of contribution dicts
        # Each dict: { index : magnitude }
        grouped = defaultdict(list)
    
        for i in range(MINVEC, MAXVEC):
             eigval = eigen[i]
             evec = evecs0[i]
     
             # 1. Take probabilities |c|² for the eigenvector
             probs = np.abs(evec) ** 2
             
             # 2. Put (prob, index) pairs and take the top N
             pairs = list(zip(probs, range(len(probs))))
             pairs.sort(reverse=True, key=lambda x: x[0])
             top=pairs[:topN] if topN>0 else pairs[:]
             
             # 3. Build structure: [eigval, (prob1, idx1), ...]
             rounded_energy = round(float(eigval), roundingLevel)  # rounding level adjustable
             # rounded_energy = eigval  # rounding level adjustable
     
             # 4. Store this eigenvector's contributions
             contrib_dict = {idx: prob for prob, idx in top}
             grouped[rounded_energy].append(contrib_dict)
             
     
        # 5. Sum + normalize contributions for each degenerate energy
        result = {}
     
        for energy, contrib_list in grouped.items():
            combined = defaultdict(float)
    
            # Sum contributions of all eigenvectors belonging to this energy
            for contrib in contrib_list:
                for idx, mag in contrib.items():
                    combined[idx] += mag
    
            # Normalize
            # norm = sum(combined.values())
            # if norm > 0:
            #     for k in combined:
            #         combined[k] /= norm
            sorted_items = sorted(
                combined.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            result[energy] = [
                (self.nspace_spatials_MCDE.returnNumLabels(idx), val)
                for idx, val in sorted_items
            ]
            # result[energy] = dict(
            #     sorted(
            #         (
            #             (self.nspace_spatials_MCDE.returnNumLabels(idx), val)
            #             for idx, val in combined.items()
            #         ),
            #         key=lambda x: x[1],
            #         reverse=True
            #     )
            # )
        
        # if ao_labels is not None:
        #     if not secondBorn:
        #         for energy in result:
        #             result[energy] = {self.nspace_spatials_MCDE.translateToAOLabels(idx,ao_labels): val for idx, val in result[energy].items()}
        #     else:
        #         for energy in result:
        #             result[energy] = {self.nspace_spatials_SB.translateToAOLabels(idx,ao_labels): val for idx, val in result[energy].items()}
        # else:
        #     for energy in result:
        #         result[energy] = {self.nspace_spatials_MCDE.returnNumLabels(idx): val for idx, val in result[energy].items()}
        return result

    

