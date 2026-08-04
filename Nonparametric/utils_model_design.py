import numpy as np

def generate_owb_design(I, J , p, seed=1):
    '''
    Generate one-way balanced (full-sib) design with I families each of size J, 
    and compute MANOVA estimator coeffecients B1.
    '''
    np.random.seed(seed)
    k = 2
    J_list=np.array([J]*I)

    n=np.sum(J_list)
    
    U0 = np.ones((n, 1))
    
    U1=np.zeros([n,I])
    sum_=0
    for i in range(I):
        U1[sum_:(sum_+J_list[i]),i]=1
        sum_=sum_+J_list[i]
                
    U2= np.eye(n)
    
    U_list = [U0,U1,U2]

    pi1 = U1@np.linalg.inv(U1.T@U1)@U1.T-U0@np.linalg.inv(U0.T@U0)@U0.T
    pi2 = np.eye(n)-U1@np.linalg.inv(U1.T@U1)@U1.T
    K_MANOVA = (n-I*J**2/n)/(I-1)
    B1_MANOVA = p*(pi1/(I-1)-pi2/(n-I))/K_MANOVA

    return U_list[1:], B1_MANOVA, J_list, n

def generate_twub_design(I_1, M, p, J=[], J2=[],seed=1):
    ''''
    Generate two-way unbalanced (full-sib half-sib) design with I_1 sires and M dams, 
    and compute MANOVA estimator coefficient B1.

    Family sizes J_{im} corresponding to the i-th sire and m-th dam are i.i.d. Unif{1,2}.    
    '''
    np.random.seed(seed)
    k = 3
    I_2 = I_1*M

    if len(J)==0:
        J2 = np.random.binomial(1, 0.5, size=(I_1, M))+1
        J = np.sum(J2,axis=1)    
        J2 = J2.reshape(-1)
    
    n=np.sum(J); 
    
    U0 = np.ones((n, 1))
    
    U1=np.zeros([n,I_1])
    sum_=0
    for i in range(I_1):
        U1[sum_:(sum_+J[i]),i]=1
        sum_=sum_+J[i]
                
    U2=np.zeros([n,len(J2)])
    sum_=0        
    for i in range(len(J2)):
        U2[sum_:(sum_+J2[i]),i]=1
        sum_=sum_+J2[i]
    
    U3 = np.eye(n)
    
    U_list = [U0,U1,U2,U3]
    
    pi_list = []
    
    for i in range(k):
        pi_list.append(U_list[i+1]@np.linalg.inv(U_list[i+1].T@U_list[i+1])@U_list[i+1].T 
                      - U_list[i]@np.linalg.inv(U_list[i].T@U_list[i])@U_list[i].T)
        
    J2_mat = J2.reshape(I_1, M)
    J1_tile = np.tile(J, (M, 1)).T
    l1 = np.sum(J2_mat**2/J1_tile)
    l2 = np.sum(J2_mat**2)/n
    l3 = np.sum(J**2)/n

    B3_MANOVA = p*pi_list[2]/(n-I_2)
    B2_MANOVA = (p*pi_list[1]-(I_2-I_1)*B3_MANOVA)/(n-l1)
    B1_MANOVA = (p*pi_list[0]-(l1-l2)*B2_MANOVA-(I_1-1)*B3_MANOVA)/(n-l3)
    B_list = [B1_MANOVA,B2_MANOVA,B3_MANOVA]

    return U_list[1:], B_list, J, J2, n