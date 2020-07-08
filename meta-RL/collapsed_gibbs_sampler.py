import numpy as np
import scipy as sci
import scipy.special as sc
import time
np.random.seed(np.int64(time.time()))

alpha = 1
num_sweeps = 1000
mu_0=0
k_0=.05
lambda_0=0.02
v_0 =2
a_0=1
b_0=1

# SPE and RPE has shape of (1,num_trials)
# RPE_SARSA has shape of (num_sessions,1) and each session data (i.e. RPE_SARSA[sei,0] has shape of (n_trials_in_session{sei},5)
# 5 indicate (s,a,r,s',a'), and a' is 0 when it is second stage action (no further action)
def sampler(PE, alpha=1, num_sweeps = 1000,mu_0=0,k_0=.05,lambda_0=0.02,v_0 =2,a_0=1,b_0=1):
    n_trials=PE.shape[1]

    alpha_0 = alpha
    [D, N] = PE.shape

    # Memory size specification
    class_id_type = 'uint16'
    max_class_id = 65535 #UNIT16 MAX INTEGER

    #
    K_plus = 1
    if class_id_type is 'uint16':
        class_id = np.zeros((N,num_sweeps),dtype=np.uint16)
    else:
        class_id = np.zeros((N, num_sweeps))
    K_record = np.zeros((num_sweeps,1))
    alpha_record = np.zeros((num_sweeps,1))
    lp_record = np.zeros((num_sweeps,1))

    # CRP first customer
    class_id[0, 0] = 1

    # precompute student-T posterior predictive distribution constants
    pc_max_ind = 1e5
    pc_gammaln_by_2 =np.linspace(1,int(pc_max_ind)+1,int(pc_max_ind))

    pc_gammaln_by_2 = sc.gammaln(pc_gammaln_by_2/2)
    pc_log_pi = np.log(np.pi)
    pc_log = np.log(np.linspace(1,int(pc_max_ind)+1,int(pc_max_ind)))

    means = np.zeros((D, max_class_id))
    sum_squares = np.zeros((D, D, max_class_id))
    inv_cov = np.zeros((D, D, max_class_id))
    log_det_cov = np.zeros((max_class_id, 1))
    counts = np.zeros((max_class_id, 1),dtype=np.uint32)
    counts[0] = 1

    y = PE[:,0]
    yyT = y*np.transpose(y)

    [lp, ldc, ic] = lp_tpp_helper(pc_max_ind, pc_gammaln_by_2, pc_log_pi, pc_log, y, 1, y, yyT, k_0, mu_0, v_0, lambda_0)

    means[:,0] = y
    sum_squares[:,:,0] = y * np.transpose(y)
    counts[0] = 1
    log_det_cov[0] = ldc
    inv_cov[:,:,0] = ic

    yyT = np.zeros((D, D, N))
    d2 = D / 2


    # pre-compute the probability of each point under the prior alone
    p_under_prior_alone = np.zeros((N, 1))

    Sigma = (lambda_0 * (k_0 + 1) / np.transpose(k_0 * (v_0 - 2 + 1)))
    v = v_0 - 2 + 1
    mu = mu_0
    try:
        log_det_Sigma = np.log(np.linalg.det(Sigma))
        inv_Sigma = np.linalg.inv(Sigma)
    except:
        log_det_Sigma = np.log(Sigma)
        inv_Sigma = 1/Sigma
    vd = v + D

    for i in range(N):
        y = PE[:,i];
        yyT[:,:,i]=y*np.transpose(y)

        if vd < pc_max_ind:
            lp = pc_gammaln_by_2[vd-1] - (pc_gammaln_by_2[v-1] + d2*pc_log[v-1] + d2*pc_log_pi) - .5*log_det_Sigma-(vd/2)*np.log(1+(1/v)*np.transpose(y-mu)*inv_Sigma*(y-mu))
        else:
            d=np.random.rand() # d??? is not exist.
            lp = sc.gammaln((v+d)/2)-(sc.gammaln(v/2) + (d/2)*np.log(v) + (d/2)*pc_log_pi)-.5*log_det_Sigma-((v+d)/2)*np.log(1+(1/v)*np.transpose(y-mu)*inv_Sigma*(y-mu))
        p_under_prior_alone[i] = lp

    # initialize timers
    time_1_obs = 0
    total_time = 0

    for sweep in range(num_sweeps):
        E_K_plus = np.mean(K_record[0:sweep+1,0])
        total_time = total_time + time_1_obs

        if sweep == 0:
            print('CRP Gibbs:: Sweep: ' + str(sweep) + '/' + str(num_sweeps))
        elif ((sweep+1) % 100) == 0:
            rem_time = (time_1_obs * .05 + 0.95 * (total_time / sweep)) * num_sweeps - total_time
            if rem_time < 0:
                rem_time = 0
            print('CRP Gibbs:: Sweep: ' + str(sweep) + '/' + str(num_sweeps) + ', Rem. Time: ' + str(rem_time) + 's, Ave. Time: ' + str((total_time / (sweep+1))) + 's, Elaps. Time: ' + str(total_time) + 's, E[K+] ' + str(E_K_plus))
        tic()

        si = 0
        if sweep==0:
            si = 1
        else:
            class_id[:,sweep] = class_id[:,sweep-1]

        for i in range(N):
            if i<si:
                'skip'
            else:
                y = PE[:,i]
                old_class_id= class_id[i,sweep]

                if old_class_id != 0:
                    if counts[old_class_id-1]==0:
                        print('')
                    counts[old_class_id-1] = counts[old_class_id-1] -1
                    if counts[old_class_id-1] == 0:
                        hits = class_id[:,sweep]>=old_class_id
                        class_id[hits,sweep] = class_id[hits,sweep]-1
                        K_plus = K_plus-1

                        hits =np.concatenate((np.linspace(0,old_class_id-2,old_class_id-2+1), \
                                             np.linspace(old_class_id, K_plus-1 + 1, K_plus-1 + 1 - old_class_id + 1))).astype(dtype = np.uint32)
                        means[:,0:K_plus] = means[:,hits]
                        means[:,K_plus] = 0
                        sum_squares[:,:,0:K_plus] = sum_squares[:,:,hits];
                        sum_squares[:,:,K_plus] = 0

                        counts[0:K_plus] = counts[hits]
                        counts[K_plus]= 0

                        log_det_cov[0:K_plus] = log_det_cov[hits]
                        log_det_cov[K_plus] = 0
                        inv_cov[:,:,0:K_plus] = inv_cov[:,:,hits]
                        inv_cov[:,:,K_plus] = 0


                    else:
                        means[:,old_class_id-1] = (1/(counts[old_class_id-1]))*((counts[old_class_id-1]+1)*means[:,old_class_id-1] - y)
                        sum_squares[:,:,old_class_id-1] = sum_squares[:,:,old_class_id-1] - yyT[:,:,i]

                # complete the CRP prior with new table prob.
                if sweep != 0:
                    pt = np.zeros((counts[0:K_plus].shape[0]+1,counts[0:K_plus].shape[1]))
                    pt[:K_plus] =counts[0:K_plus]
                    pt[pt.shape[0]-1] = alpha
                    prior = pt/(N - 1 + alpha)
                else:
                    pt = np.zeros((counts[0:K_plus].shape[0]+1,counts[0:K_plus].shape[1]))
                    pt[:K_plus] =counts[0:K_plus]
                    pt[pt.shape[0]-1] = alpha
                    prior = pt/(i+1 - 1 + alpha)

                likelihood = np.zeros((np.max(prior.shape),1))
                # for ell in range(K_plus):
                for ell in np.linspace(0,K_plus-1,K_plus-0).astype(dtype=np.int16).tolist():
                    # get the class ids of the points sitting at table l
                    n = counts[ell]
                    m_Y = means[:,ell]
                    mu_n = k_0/(k_0+n)*mu_0 + n/(k_0+n)*m_Y
                    k_n = k_0+n
                    v_n = v_0+n

                    # set up variables for Gelman's formulation of the Student T distribution
                    v = v_n-2+1
                    mu = mu_n

                    if old_class_id != 0:
                        if (old_class_id-1) == ell:
                            S = (sum_squares[:,:,ell] - n*m_Y*np.transpose(m_Y))
                            zm_Y = m_Y-mu_0
                            lambda_n = lambda_0 + S  + k_0*n/(k_0+n)*(zm_Y)*np.transpose(zm_Y)
                            Sigma = np.transpose(lambda_n*(k_n+1)/(k_n*(v_n-2+1)))

                            old_class_log_det_Sigma = log_det_cov[old_class_id-1]
                            old_class_inv_Sigma = inv_cov[:,:,old_class_id-1]

                            try:
                                log_det_Sigma = np.log(np.linalg.det(Sigma))
                                inv_Sigma = np.linalg.inv(Sigma)
                            except:
                                log_det_Sigma = np.log(Sigma)
                                inv_Sigma = 1 / Sigma

                            log_det_cov[old_class_id-1] = log_det_Sigma
                            inv_cov[:,:,old_class_id-1] = inv_Sigma
                        else:
                            log_det_Sigma = log_det_cov[ell]
                            inv_Sigma = inv_cov[:,:,ell]
                    else:
                        # this case is the first sweep through the data
                        S = sum_squares[:,:,ell] - n*m_Y*np.transpose(m_Y)
                        zm_Y = m_Y-mu_0
                        lambda_n = lambda_0 + S  + k_0*n/(k_0+n)*(zm_Y)*np.transpose(zm_Y)
                        Sigma = np.transpose(lambda_n*(k_n+1)/(k_n*(v_n-2+1)))

                        try:
                            log_det_Sigma = np.log(np.linalg.det(Sigma))
                            inv_Sigma = np.linalg.inv(Sigma)
                        except:
                            log_det_Sigma = np.log(Sigma)
                            inv_Sigma = 1 / Sigma

                        log_det_cov[ell] = log_det_Sigma
                        inv_cov[:,:,ell] = inv_Sigma
                    vd = v + D
                    if vd < pc_max_ind:
                        lp = pc_gammaln_by_2[vd-1] - (pc_gammaln_by_2[v-1] + d2 * pc_log[v-1] + d2 * pc_log_pi) - .5 * log_det_Sigma\
                             - (vd / 2) * np.log(1 + (1 / v) * np.transpose(y - mu)*inv_Sigma*(y-mu))
                    else:
                        lp = sc.gammaln((v + d) / 2) - (sc.gammaln(v / 2) + (d / 2) * np.log(v) + (d / 2) * pc_log_pi) - .5 * log_det_Sigma - ((v + d) / 2) * np.log(1 + (1 / v) * np.transpose(y - mu)*inv_Sigma*(y-mu))

                    likelihood[ell] = lp

                likelihood[K_plus-1 + 1] = p_under_prior_alone[i]

                likelihood = np.exp(likelihood - np.max(likelihood))
                likelihood = likelihood / np.sum(likelihood)
                # compute the posterior over seating assignment for datum i
                posterior = np.multiply(prior,likelihood) # this is actually a proportionality
                # normalize the posterior
                posterior = posterior / np.sum(posterior)

                # pick the new table
                cdf = np.cumsum(posterior)
                rn = np.random.rand()

                new_class_id = np.where((cdf>rn))[0][0]+1

                if new_class_id > max_class_id:
                    print('K^plus has exceeded the maximum value of ' + class_id_type )
                    return

                counts[new_class_id-1] = counts[new_class_id-1] + 1

                means[:, new_class_id-1] = means[:, new_class_id-1]+ (1 / (counts[new_class_id-1])) * (y - means[:, new_class_id-1])
                sum_squares[:,:, new_class_id-1] = sum_squares[:,:, new_class_id-1] + yyT[:,:, i]

                if new_class_id == K_plus + 1:
                    K_plus = K_plus + 1

                if old_class_id == new_class_id:
                    log_det_cov[old_class_id-1] = old_class_log_det_Sigma
                    inv_cov[:,:,old_class_id-1] = old_class_inv_Sigma
                else:
                    # the point changed tables which means that the matrix inverse
                    # sitting in the old_class_id slot is appropriate but that the
                    # new table matrix inverse needs to be updated
                    n = counts[new_class_id-1]
                    #             if n~=0
                    m_Y = means[:,new_class_id-1]
                    k_n = k_0+n
                    v_n = v_0+n

                    # set up variables for Gelman's formulation of the Student T
                    # distribution
                    S = (sum_squares[:,:,new_class_id-1] - n*m_Y*np.transpose(m_Y))
                    zm_Y = m_Y-mu_0
                    lambda_n = lambda_0 + S  + k_0*n/(k_0+n)*(zm_Y)*np.transpose(zm_Y)
                    Sigma = np.transpose(lambda_n*(k_n+1)/(k_n*(v_n-2+1)))

                    try:
                        log_det_cov[new_class_id-1] = np.log(np.linalg.det(Sigma))
                        inv_cov[:,:,new_class_id-1] = np.linalg.inv(Sigma)
                    except:
                        log_det_cov[new_class_id-1] = np.log(Sigma)
                        inv_cov[:,:,new_class_id-1] = 1 / Sigma

                # record the new table
                class_id[i, sweep] = new_class_id

        lZ = lp_mvniw(class_id[:, sweep], PE, mu_0, k_0, v_0, lambda_0)
        lp = lp_crp(class_id[:, sweep], alpha) # -  gamlike([a_0 b_0], alpha);

        lp = lp + lZ

        nu = np.random.beta(alpha,N)+np.finfo(float).eps
        if sweep+1 > 50:
            alpha = 1/np.random.gamma(a_0+K_plus-1,b_0-np.log(nu))

        # record the current parameters values
        K_record[sweep] = K_plus
        alpha_record[sweep] = alpha
        lp_record[sweep] = lp

        time_1_obs = toc()

    return [class_id, K_record, lp_record, alpha_record]
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    # if 'startTime_for_tictoc' in globals():
        # print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    # else:
        # print("Toc: start time not set")
    return time.time() - startTime_for_tictoc

def lp_mvniw(class_id, training_data, mu_0, k_0, v_0, lambda_0):
    lZ = 0
    d = training_data.shape[0]

    if d > v_0:
        print('v_0 must be equal to or larger than the dimension of the data')
        return

    K_plus = np.max(np.unique(class_id).shape)

    for l in np.linspace(0,K_plus-1,K_plus-0).astype(dtype=np.int16).tolist():
        # get the class ids of the points sitting at table l
        hits_= (class_id-1)==l
        hits = np.zeros(hits_.shape)
        hits[np.where(hits_)[0]] = 1

        # how many are sitting at table l?
        n = np.sum(hits)
        # get those points
        Y = training_data[:,np.where(hits_)[0]]
        if n!= 0:
            mean_Y = np.mean(Y,axis=1)
            k_n = k_0+n
            v_n = v_0+n

            S = np.transpose(np.cov(Y))*(n-1)
            lambda_n = lambda_0 + S  + k_0*n/(k_0+n)*(mean_Y-mu_0)*np.transpose(mean_Y-mu_0)
        else:
            print('Should always have one element')
            return
        if n % 2 !=0:
            ls = 0
            for j in range(int(d)):
                ls = ls + sc.gammaln((v_n+1-j+1)/2) - sc.gammaln((v_0+1-j+1)/2)
        else:
            ls = 0
            for j in range(int(d)):
                for ii in range(int(np.floor(n/2))): #=1:floor(n/2)
                    ls = ls + np.log((v_n+1-j-1)/2-ii-1)
        try:
            lZ = lZ - n * d / 2 * np.log(2 * np.pi) + d / 2 * (np.log(k_0) - np.log(k_n)) +  d / 2 * (v_n - v_0) * np.log(2) + v_0 / 2 * np.log(np.linalg.det(lambda_0)) - v_n / 2 * np.log(np.linalg.det(lambda_n)) + ls
        except:
            lZ = lZ - n * d / 2 * np.log(2 * np.pi) + d / 2 * (np.log(k_0) - np.log(k_n)) +  d / 2 * (v_n - v_0) * np.log(2) + v_0 / 2 * np.log(lambda_0) - v_n / 2 * np.log(lambda_n) + ls
    return lZ

def lp_crp(c, alpha):
    table_identifier = np.unique(c)
    K_plus = np.max(table_identifier.shape)
    N = np.max(c.shape)
    m_k = np.zeros((K_plus,1))
    for k in range(K_plus):
        m_k[k] = np.sum(c == table_identifier[k])
    foo = sc.gammaln(m_k-1)
    try:
        foo[(m_k-1)==0]=0
    except:
        print('')
    lp = K_plus * np.log(alpha) + np.sum(foo) + sc.gammaln(alpha) - sc.gammaln(N + alpha)
    return lp

def lp_tpp_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,n,m_Y,SS,k_0,mu_0,v_0,lambda_0):
    d = y.shape[0]
    if n != 1:
        mu_n = k_0/(k_0+n)*mu_0 + n/(k_0+n)*m_Y
        k_n = k_0+n
        v_n = v_0+n
        S = (SS - n*m_Y*np.transpose(m_Y))
        zm_Y = m_Y-mu_0
        lambda_n = lambda_0 + S  + k_0*n/(k_0+n)*(zm_Y)*np.transpose(zm_Y)
    else:
        mu_n = mu_0
        k_n = k_0
        v_n = v_0
        lambda_n = lambda_0

    Sigma = lambda_n * (k_n + 1) / (k_n * (v_n - 2 + 1))
    v = v_n-2+1
    mu = mu_n

    # in general use, check lp_tpp_helper with more arguements
    try:
        log_det_Sigma = np.log(np.linalg.det(Sigma))
    except: # if the Sigma has 1x1 shape (general cases)
        log_det_Sigma = np.log(Sigma)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except:# if the Sigma has 1x1 shape (general cases)
        inv_Sigma = 1/Sigma

    # if the values have been precomputed use them
    vd = v + d
    if vd<pc_max_ind:
        d2 = d/2
        lp = pc_gammaln_by_2[vd-1] - (pc_gammaln_by_2[v-1] + d2*pc_log[v-1] + d2*pc_log_pi) - .5*log_det_Sigma-(vd/2)*np.log(1+(1/v)*np.transpose(y-mu)*inv_Sigma*(y-mu))
    else:
        lp = sc.gammaln((v+d)/2)-(sc.gammaln(v/2) + (d/2)*np.log(v) + (d/2)*pc_log_pi)-.5*log_det_Sigma-((v+d)/2)*np.log(1+(1/v)*np.transpose(y-mu)*inv_Sigma*(y-mu))
    return [lp, log_det_Sigma, inv_Sigma]