/* minunit.h - https://jera.com/techinfo/jtns/jtn002 */
#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do { char *message = test(); tests_run++; \
                               if (message) return message; } while (0)
#define mu_assert_close(msg, a, b) mu_assert(msg, fabs((a) - (b)) < EPSILON)
extern int tests_run;
