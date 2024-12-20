from itertools import product
import re


class Predicate:
    CONSTANTS = {"a", "b", "c", "d", "e"}

    def __init__(self, name, args, negated=False):
        self.name = name
        self.args = args
        self.negated = negated

    def __neg__(self):
        return Predicate(self.name, self.args, not self.negated)

    def __str__(self):
        return f"{'¬' if self.negated else ''}{self.name}({', '.join(map(str, self.args))})"

    def __eq__(self, other):
        return isinstance(other,
                          Predicate) and self.name == other.name and self.args == other.args and self.negated == other.negated

    def __hash__(self):
        return hash((self.name, tuple(self.args), self.negated))

    @staticmethod
    def occurs_check(variable, term):
        if variable == term:
            return True
        if "f(" in term:  # Assuming terms like f(x)
            inner_args = term[2:-1].split(",")  # Extract arguments of the term
            return any(Predicate.occurs_check(variable, arg.strip()) for arg in inner_args)
        return False

    @staticmethod
    def unify(term1, term2):
        if term1 == term2:
            return {}  # Identical terms unify
        # Variable unification
        if not Predicate.occurs_check(term1, term2) and not Predicate.occurs_check(term2, term1):
            if not Predicate.is_constant(term1):
                return {term1: term2}
            if not Predicate.is_constant(term2):
                return {term2: term1}
        # Both are constants and different
        if Predicate.is_constant(term1) and Predicate.is_constant(term2):
            return None
        # Cannot unify otherwise
        return None

    @staticmethod
    def is_constant(term):
        return term in Predicate.CONSTANTS

    def can_unify(self, other):
        if Predicate.is_constant(self) and Predicate.is_constant(other):
            if self.name != other.name or len(self.args) != len(other.args):
                return False  # Different constant
        # Unify arguments
        for arg1, arg2 in zip(self.args, other.args):
            if Predicate.unify(arg1, arg2) is None:
                return False
        return True


class Clause:
    def __init__(self, literals):
        self.literals = set(literals)

    def __str__(self):
        return " ∨ ".join(map(str, self.literals))

    def __eq__(self, other):
        return isinstance(other, Clause) and self.literals == other.literals

    def __hash__(self):
        return hash(frozenset(self.literals))

    def __repr__(self):
        return f"Clause({self.literals})"

    def resolve(self, other):
        for literal in self.literals:
            for other_literal in other.literals:
                if (
                        literal.name == other_literal.name
                        and literal.negated != other_literal.negated
                        and literal.can_unify(other_literal)
                ):
                    # Perform resolution
                    new_literals = self.literals.union(other.literals)
                    new_literals.remove(literal)
                    new_literals.remove(other_literal)
                    return Clause(new_literals)
        return None


# Given an CNF, use resolution method to test if it is unsatisfiable
def resolution(clauses):
    new = set()
    while True:
        n = len(clauses)
        clause_list = list(clauses)  # Convert the set to a list for indexing
        pairs = [(clause_list[i], clause_list[j]) for i in range(n) for j in range(i + 1, n)]

        for (ci, cj) in pairs:
            resolvent = ci.resolve(cj)
            if resolvent is not None:
                if not resolvent.literals:  # Empty clause
                    return True
                new.add(resolvent)

        if new.issubset(clauses):
            return False

        clauses.update(new)


def parse_predicate(predicate_str):
    # Remove outer brackets if they enclose the entire predicate
    predicate_str = predicate_str.strip()
    if predicate_str.startswith("(") and predicate_str.endswith(")"):
        # Ensure the parentheses match the entire predicate
        if validate_parentheses(predicate_str[1:-1]):
            predicate_str = predicate_str[1:-1]

    # Check if the predicate is negated
    negated = predicate_str.startswith("¬")
    # Remove the negation symbol if present
    predicate_str = predicate_str[1:] if negated else predicate_str

    # Extract predicate name and arguments
    match = re.match(r"(\w+)\((.*?)\)", predicate_str)
    if not match:
        raise ValueError(f"Invalid predicate format: {predicate_str}")
    name = match.group(1)
    args = match.group(2).split(",")  # Split arguments by comma

    # Return a Predicate object
    return Predicate(name.strip(), [arg.strip() for arg in args], negated)


def parse_clause(clause_str):
    if not validate_parentheses(clause_str):
        raise ValueError("Parentheses in the CNF string are unbalanced or incorrectly ordered.")

    # Split the clause by the `∨` operator
    predicates = clause_str.split("∨")

    # Fix parentheses for each predicate
    fixed_predicates = [fix_parentheses(pred.strip()) for pred in predicates]

    # Parse each predicate into a Literal or Predicate object
    literals = [parse_predicate(pred) for pred in fixed_predicates]

    # Return a Clause object containing all literals
    return Clause(literals)


def fix_parentheses(predicate_str):
    while not validate_parentheses(predicate_str):
        # Remove the first '(' if the first character is '('
        if predicate_str.startswith('('):
            predicate_str = predicate_str[1:]
        # Remove the last ')' if the last character is ')'
        elif predicate_str.endswith(')'):
            predicate_str = predicate_str[:-1]
        else:
            break  # Stop if no obvious fix is possible
    return predicate_str


def parse_clauses(clauses_list):
    # Parse each clause string into a Clause object
    clauses = {parse_clause(clause_str.strip()) for clause_str in clauses_list}
    return clauses


def parse_cnf(cnf_str):
    # Tokenize the CNF string
    tokens = tokenize_cnf(cnf_str)
    # Parse tokens into clauses
    clauses = parse_tokens(tokens)
    return clauses


def tokenize_cnf(cnf_str):
    # Validate parentheses
    if not validate_parentheses(cnf_str):
        raise ValueError("Parentheses in the CNF string are unbalanced or incorrectly ordered.")

    tokens = []
    current_token = []
    stack = 0  # Keep track of nested parentheses

    for char in cnf_str:
        if char == '(':
            stack += 1
            current_token.append(char)
        elif char == ')':
            stack -= 1
            if stack < 0:  # Unbalanced parenthesis detected
                raise ValueError("Parentheses are incorrectly balanced.")
            current_token.append(char)
        elif char in {'∧', '∨'} and stack == 0:
            # Complete the current token before processing the operator
            if current_token:
                tokens.append("".join(current_token).strip())
                current_token = []
            tokens.append(char)
        else:
            current_token.append(char)

    # Add the last token, if any
    if current_token:
        tokens.append("".join(current_token).strip())
    if stack != 0:  # Unbalanced parenthesis at the end
        raise ValueError("Parentheses are incorrectly balanced.")
    return tokens


def validate_parentheses(expression):
    stack = 0
    for char in expression:
        if char == '(':
            stack += 1
        elif char == ')':
            stack -= 1
            if stack < 0:
                return False

    return stack == 0


def parse_tokens(tokens):
    clauses = set()
    clause_tokens = []
    for token in tokens:
        if token == '∧':  # End of current clause, start a new one
            if clause_tokens:
                clauses.add(parse_clause("".join(clause_tokens)))
                clause_tokens = []
        else:
            clause_tokens.append(token)
    if clause_tokens:  # Add the last clause
        clauses.add(parse_clause("".join(clause_tokens)))
    return clauses


def tokenize_fol(fol_str):
    token_pattern = r'¬|∧|∨|->|<->|\(|\)|[a-zA-Z_][a-zA-Z0-9_]*\(.*?\)|[a-zA-Z_][a-zA-Z0-9_]*'
    if not validate_parentheses(fol_str):
        raise ValueError("Parentheses in the CNF string are unbalanced or incorrectly ordered.")

    # Tokenize the CNF string
    tokens = re.findall(token_pattern, fol_str)
    return tokens


# Helper to find a full subexpression (handles nested parentheses)
def find_subexpression(tokens, start_idx):
    subexpr = []
    open_parens = 0

    for i in range(start_idx, len(tokens)):
        token = tokens[i]
        subexpr.append(token)
        if token == '(':
            open_parens += 1
        elif token == ')':
            open_parens -= 1
            if open_parens == 0:  # All parentheses are balanced
                return ''.join(subexpr), i
    raise ValueError("Unmatched parentheses in the expression!")


# Helper function to eliminate implications (A -> B becomes ¬A ∨ B) //NOT WORKING YET
def eliminate_implies(tokens):
    result = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == '->':
            # Replace A -> B with ¬A ∨ B
            left = result.pop()  # Pop the left-hand side (A)
            right = tokens[i + 1]  # The right-hand side (B)
            if left == ')':
                sub_expr = []
                sub_expr.append(')')
                counter = 1
                # Reconstruct the subexpression within the parentheses
                while result:
                    elem = result.pop()
                    sub_expr.append(elem)
                    if elem == '(':
                        counter -= 1
                    elif elem == ')':
                        counter += 1
                    if counter == 0:
                        break
                if counter != 0:
                    raise ValueError("Unmatched parentheses in the expression!")

                # Reverse the subexpression to restore the correct order
                sub_expr = sub_expr[::-1]

                # Add `¬` to the entire subexpression
                result.append('¬')
                result.extend(sub_expr)
                result.extend(['∨', right])
            else:
                # Wrap in parentheses to ensure correct precedence
                result.extend(['¬', left, '∨', right])
            i += 1  # Skip the right operand (B)
        # elif token == '<->':
        #     left = result.pop()
        #     right = tokens[i + 1]
        #     # Handle nested expressions for `left`
        #     if left == ')':
        #         sub_expr = []
        #         sub_expr.append(')')
        #         counter = 1
        #         while result:
        #             elem = result.pop()
        #             sub_expr.append(elem)
        #             if elem == '(':
        #                 counter -= 1
        #             elif elem == ')':
        #                 counter += 1
        #             if counter == 0:
        #                 break
        #         if counter != 0:
        #             print(sub_expr)
        #             raise ValueError("Unmatched parentheses in the expression!")

        #         # Reverse sub_expr to restore original order
        #         sub_expr = sub_expr[::-1]
        #         left = sub_expr

        #     # Handle nested expressions for `right`
        #     if right == '(':
        #         sub_expr = []
        #         sub_expr.append('(')
        #         counter = 1
        #         j = i + 2
        #         while j < len(tokens):
        #             elem = tokens[j]
        #             sub_expr.append(elem)
        #             if elem == '(':
        #                 counter += 1
        #             elif elem == ')':
        #                 counter -= 1
        #             if counter == 0:
        #                 break
        #             j += 1
        #         if counter != 0:
        #             raise ValueError("Unmatched parentheses in the expression!")
        #         right = sub_expr
        #         i = j  # Move index to the end of the processed subexpression

        #     # Append the transformation
        #     result.append('(')
        #     result.append('¬')
        #     result.extend(left)
        #     result.append('∨')
        #     result.extend(right)
        #     result.append(')')
        #     result.append('∧')
        #     result.append('(')
        #     result.append('¬')
        #     result.extend(right)
        #     result.append('∨')
        #     result.extend(left)
        #     result.append(')')
        #     i += 1  # Skip the right operand (B)

        else:
            result.append(token)

        i += 1
    return result


def is_complex_expression(expr):
    return isinstance(expr, list)


def eliminate_equivalence(tokens):
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '<->':
            left = result.pop()
            right = tokens[i + 1]
            # Handle nested expressions for `left`
            if left == ')':
                sub_expr = []
                sub_expr.append(')')
                counter = 1
                while result:
                    elem = result.pop()
                    sub_expr.append(elem)
                    if elem == '(':
                        counter -= 1
                    elif elem == ')':
                        counter += 1
                    if counter == 0:
                        break
                if counter != 0:
                    print(sub_expr)
                    raise ValueError("Unmatched parentheses in the expression!")

                # Reverse sub_expr to restore original order
                sub_expr = sub_expr[::-1]
                left = sub_expr

            # Handle nested expressions for `right`
            if right == '(':
                sub_expr = []
                sub_expr.append('(')
                counter = 1
                j = i + 2
                while j < len(tokens):
                    elem = tokens[j]
                    sub_expr.append(elem)
                    if elem == '(':
                        counter += 1
                    elif elem == ')':
                        counter -= 1
                    if counter == 0:
                        break
                    j += 1
                if counter != 0:
                    raise ValueError("Unmatched parentheses in the expression!")
                right = sub_expr
                i = j  # Move index to the end of the processed subexpression

            # Append the transformation
            result.append('(')
            result.append('¬')
            if (is_complex_expression(left)):
                result.extend(left)
            else:
                result.append(left)
            result.append('∨')
            if (is_complex_expression(right)):
                result.extend(right)
            else:
                result.append(right)
            result.append(')')
            result.append('∧')
            result.append('(')
            result.append('¬')
            if (is_complex_expression(right)):
                result.extend(right)
            else:
                result.append(right)
            result.append('∨')
            if (is_complex_expression(left)):
                result.extend(left)
            else:
                result.append(left)
            result.append(')')
            i += 1  # Skip the right operand (B)

        else:
            result.append(token)

        i += 1
    return result


def push_negation(tokens):  #Not Used
    result = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == '¬':
            # Handle negation
            inner = tokens[i + 1]

            if inner == '(':
                # Extract subexpression within parentheses
                sub_expr = []
                counter = 1
                i += 2  # Skip '¬' and '('
                while i < len(tokens):
                    if tokens[i] == '(':
                        counter += 1
                    elif tokens[i] == ')':
                        counter -= 1
                    if counter == 0:
                        break
                    sub_expr.append(tokens[i])
                    i += 1

                print("sub = ", sub_expr[1])
                # Apply De Morgan's laws to the subexpression
                if sub_expr[1] == '∧':
                    # ¬(A ∧ B) -> ¬A ∨ ¬B
                    left = ['¬'] + [sub_expr[0]]
                    right = ['¬'] + [sub_expr[2]]
                    result.extend(['('] + push_negation(left) + ['∨'] + push_negation(right) + [')'])
                elif sub_expr[1] == 'V':
                    # ¬(A ∨ B) -> ¬A ∧ ¬B
                    left = ['¬'] + [sub_expr[0]]
                    right = ['¬'] + [sub_expr[2]]
                    result.extend(['('] + push_negation(left) + ['∧'] + push_negation(right) + [')'])
                else:
                    # Negate the entire subexpression
                    negated_sub_expr = push_negation(['¬'] + sub_expr)
                    result.extend(['('] + negated_sub_expr + [')'])

            elif inner == '¬':
                # Double negation: ¬¬A -> A
                result.append(tokens[i + 2])
                i += 2  # Skip '¬¬'

            else:
                # Negate a single variable or predicate
                result.extend(['¬', inner])
                i += 1

        else:
            # Directly append other tokens
            result.append(token)

        i += 1

    return result


# Version 3
# Conversion to NNF. Assume implications are already eliminated.
def push_negations_inward(tokens):
    if isinstance(tokens, str):  # Base case: a literal or a predicate
        return tokens

    if tokens[0] == '¬':  # Handle negation
        inner = tokens[1]
        if inner == '¬':  # If the second element is also a '¬'
            return push_negations_inward(tokens[2])
        if isinstance(inner, list):  # If negating a compound expression
            if inner[0] == '∧':  # ¬(A ∧ B) → ¬A ∨ ¬B
                return ['∨', push_negations_inward(['¬', inner[1]]), push_negations_inward(['¬', inner[2]])]
            elif inner[0] == '∨':  # ¬(A ∨ B) → ¬A ∧ ¬B
                return ['∧', push_negations_inward(['¬', inner[1]]), push_negations_inward(['¬', inner[2]])]
            elif inner[0] == '¬':  # Double negation: ¬¬A → A
                return push_negations_inward(inner[1])
        # Negation of a literal or unhandled structure
        return ['¬', push_negations_inward(inner)]

    # Recurse for compound expressions
    if tokens[0] in ('∧', '∨'):
        return [tokens[0], push_negations_inward(tokens[1]), push_negations_inward(tokens[2])]

    return tokens


def distribute_or_over_and(formula):
    if not isinstance(formula, list) or formula[0] not in ('∧', '∨'):  # Base case
        return formula

    if formula[0] == '∨':
        left, right = formula[1], formula[2]

        if isinstance(left, list) and left[0] == '∧':  # (A ∧ B) ∨ C → (A ∨ C) ∧ (B ∨ C)
            return ['∧', distribute_or_over_and(['∨', left[1], right]),
                    distribute_or_over_and(['∨', left[2], right])]

        if isinstance(right, list) and right[0] == '∧':  # A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
            return ['∧', distribute_or_over_and(['∨', left, right[1]]),
                    distribute_or_over_and(['∨', left, right[2]])]

    if formula[0] == '∧':  # Process conjunctions recursively
        return ['∧', distribute_or_over_and(formula[1]), distribute_or_over_and(formula[2])]

    return formula  # Return the simplified formula


def convert_to_cnf_inner(formula):
    if isinstance(formula, str):  # Base case: single variable or predicate
        return formula

    if formula[0] == '¬':  # Negation
        return ['¬', convert_to_cnf_inner(formula[1])]

    if formula[0] in ('∧', '∨'):  # Disjunction or conjunction
        return distribute_or_over_and([formula[0], convert_to_cnf_inner(formula[1]), convert_to_cnf_inner(formula[2])])

    return formula


# Converts a flat list of tokens into a structured tree based on logical operators.
def build_expression_tree(tokens):
    # If there's only one token, return it directly (base case)
    if len(tokens) == 1:
        return tokens[0]  # Single literal or negation

    # Look for logical operators in order of precedence: ∧ first, then ∨
    for op in ['∧', '∨']:
        if op in tokens:
            idx = tokens.index(op)  # Find the operator
            # Group operator with its left and right operands
            return [op, build_expression_tree(tokens[:idx]), build_expression_tree(tokens[idx + 1:])]

    # If no operators found, return tokens as-is (may happen for literals or negations)
    return tokens


# convert flat list of token to nested structure
def parse_nested(tokens):
    stack = []  # Stack to track nested structures
    current = []  # Current level of tokens

    for token in tokens:
        if token == '(':
            # Start a new subexpression
            stack.append(current)
            current = []
        elif token == ')':
            # End of the current subexpression
            if stack:
                last = stack.pop()
                last.append(build_expression_tree(current))  # Group the subexpression
                current = last
        else:
            # Add token to the current structure
            current.append(token)

    if len(stack) > 0:
        print("stack:", stack)
        raise ValueError("Unmatched parentheses in the token list.")

    # Process the top-level expression
    return build_expression_tree(current)


# convert a nested NNF structure into a flat tokenized format
def tokenize_human_read_nnf(tokens):
    if isinstance(tokens, str):  # Base case: a literal or predicate
        return [tokens]

    if tokens[0] == '¬':  # Negation
        return ['¬'] + tokenize_human_read_nnf(tokens[1])

    if tokens[0] in ('∧', '∨'):  # Conjunction or disjunction
        # Recursively process left and right operands
        left = tokenize_human_read_nnf(tokens[1])
        right = tokenize_human_read_nnf(tokens[2])
        return ['('] + left + [tokens[0]] + right + [')']

    return []  # Handle unexpected cases (shouldn't happen with valid input)


# Helper func that convert FOL to NNF.
def to_nnf(fol_str):
    # Tokenize the FOL expression
    tokens = tokenize_fol(fol_str)

    tokens = eliminate_implies(tokens)
    tokens = eliminate_equivalence(tokens)

    tokens = parse_nested(tokens)  # change to nested structure
    tokens = push_negations_inward(tokens)  # do negation
    tokens = tokenize_human_read_nnf(tokens)  # change back to flat list token

    return ''.join(tokens)


# Convert FOL to CNF
def to_cnf(fol_str):
    # Tokenize the FOL expression
    tokens = tokenize_fol(fol_str)

    # Step 1: Eliminate implications (A -> B becomes ¬A ∨ B)
    tokens = eliminate_implies(tokens)
    tokens = eliminate_equivalence(tokens)

    # Step 2: De Morgan's law
    tokens = parse_nested(tokens)
    tokens = push_negations_inward(tokens)
    tokens = convert_to_cnf_inner(tokens)
    tokens = simplify_parentheses(tokens)
    tokens = tokenize_human_read_nnf(tokens)
    # tokens = push_negation(tokens)

    # Return the CNF string
    return ''.join(tokens)


def tokens_to_readable(tokens):
    result = []
    for token in tokens:
        if token == '∧':
            result.append(' ∧ ')
        elif token == '∨':
            result.append(' ∨ ')
        elif token == '¬':
            result.append('¬')
        elif token == '->':
            result.append(' -> ')
        elif token == '<->':
            result.append(' <-> ')
        elif token == '(' or token == ')':
            result.append(token)
        else:
            result.append(token)
    return ''.join(result)


def simplify_parentheses(tokens):
    def process(tokens):
        if isinstance(tokens, str):  # Base case: single token
            return tokens

        if isinstance(tokens, list):
            if tokens[0] == '∧':  # Conjunction
                # Join clauses with "∧" and remove unnecessary parentheses
                left = process(tokens[1])
                right = process(tokens[2])
                return f"{left} ∧ {right}"
            elif tokens[0] == '∨':  # Disjunction
                # Keep parentheses for disjunction
                left = process(tokens[1])
                right = process(tokens[2])
                return f"({left} ∨ {right})"
            elif tokens[0] == '¬':  # Negation
                # Apply negation directly to the inner expression
                inner = process(tokens[1])
                return f"¬{inner}"

        return " ".join(map(process, tokens))  # Recurse on nested structures

    # Start processing the tokenized formula
    simplified = process(tokens)
    return simplified


if __name__ == '__main__':  #Test Cases, NOT THE MAIN APP
    # ((A -> B) ∧ (C ∨ ¬D)) -> (E ∧ F) to ¬((¬A ∨ B) ∧ (C ∨ ¬D)) ∨ (E ∧ F)
    # tokens = tokenize_fol("¬(((A -> B) ∧ ¬C) -> D)")
    # tokens = eliminate_implications(tokens)
    # tokens = ['¬','(', 'A','∧', 'B',')']
    # print (push_negation(tokens))
    print("\nThis is only for testing purposes, run main.py for main application\n\n")

    # Test Case 1: ((A -> B) ∧ (C ∨ ¬D)) -> (E ∧ F)
    tokens1 = tokenize_fol('((A -> B) ∧ (C ∨ ¬D)) -> (E ∧ F)')
    tokens1_after_elimination_implies = eliminate_implies(tokens1)
    print("\nTest Case 1:")
    print("Original Tokens:", tokens1)
    print("After Eliminating Implications:", tokens1_after_elimination_implies)
    nested_tokens1 = parse_nested(tokens1_after_elimination_implies)  # Convert to nested structure
    nnf1 = push_negations_inward(nested_tokens1)
    print("1. Nested Tokens:", nested_tokens1)
    print("1. NNF Result:", nnf1)
    simplify_nnf1 = simplify_parentheses(nnf1)
    print("1. Human read nnf:", tokenize_human_read_nnf(simplify_nnf1))
    print("1. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf1)))

    # Test Case 2: (A <-> B) -> C
    tokens2 = tokenize_fol('(f(a) ∧ f(b)) <-> f(x)')
    tokens2_after_elimination_implies = eliminate_implies(tokens2)
    tokens2_after_elimination_equivalence = eliminate_equivalence(tokens2_after_elimination_implies)
    print("\nTest Case 2:")
    print("Original Tokens:", tokens2)
    print("After Eliminating Implications:", tokens2_after_elimination_equivalence)
    nested_tokens2 = parse_nested(tokens2_after_elimination_equivalence)  # Convert to nested structure
    nnf2 = push_negations_inward(nested_tokens2)
    print("2. Nested Tokens:", nested_tokens2)
    print("2. NNF Result:", nnf2)
    simplify_nnf2 = simplify_parentheses(nnf2)
    print("2. Human read nnf:", tokenize_human_read_nnf(simplify_nnf2))
    print("2. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf2)))

    # Test Case 3: (P(x) -> Q(x)) ∧ (¬R(y) -> S(y))
    tokens3 = tokenize_fol('(P(x) -> Q(x)) ∧ (¬R(y) -> S(y))')
    tokens3_after_elimination_implies = eliminate_implies(tokens3)
    print("\nTest Case 3:")
    print("Original Tokens:", tokens3)
    print("After Eliminating Implications:", tokens3_after_elimination_implies)
    nested_tokens3 = parse_nested(tokens3_after_elimination_implies)  # Convert to nested structure
    nnf3 = push_negations_inward(nested_tokens3)
    print("3. Nested Tokens:", nested_tokens3)
    print("3. NNF Result:", nnf3)
    simplify_nnf3 = simplify_parentheses(nnf3)
    print("3. Human read nnf:", tokenize_human_read_nnf(simplify_nnf3))
    print("3. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf3)))

    # Test Case 4: ((A ∨ B) -> (C ∧ D)) <-> (E -> F)
    tokens4 = tokenize_fol('((A ∨ B) -> (C ∧ D)) <-> (E -> F)')
    tokens4_after_elimination_implies = eliminate_implies(tokens4)
    tokens4_after_elimination_equivalence = eliminate_equivalence(tokens4_after_elimination_implies)
    print("\nTest Case 4:")
    print("Original Tokens:", tokens4)
    print("After Eliminating Implications:", tokens4_after_elimination_equivalence)
    nested_tokens4 = parse_nested(tokens4_after_elimination_equivalence)  # Convert to nested structure
    nnf4 = push_negations_inward(nested_tokens4)
    print("4. Nested Tokens:", nested_tokens4)
    print("4. NNF Result:", nnf4)
    simplify_nnf4 = simplify_parentheses(nnf4)
    print("4. Human read nnf:", tokenize_human_read_nnf(simplify_nnf4))
    print("4. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf4)))

    # Test Case 5: (A -> (B ∧ C)) ∨ (D <-> E)
    tokens5 = tokenize_fol('(A -> (B ∧ C)) ∨ (D <-> E)')
    tokens5_after_elimination_implies = eliminate_implies(tokens5)
    tokens5_after_elimination_equivalence = eliminate_equivalence(tokens5_after_elimination_implies)
    print("\nTest Case 5:")
    print("Original Tokens:", tokens5)
    print("After Eliminating Implications:", tokens5_after_elimination_equivalence)
    nested_tokens5 = parse_nested(tokens5_after_elimination_equivalence)  # Convert to nested structure
    nnf5 = push_negations_inward(nested_tokens5)
    print("5. Nested Tokens:", nested_tokens5)
    print("5. NNF Result:", nnf5)
    simplify_nnf5 = simplify_parentheses(nnf5)
    print("5. Human read nnf:", tokenize_human_read_nnf(simplify_nnf5))
    print("5. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf5)))

    # Test Case 6: ((A ∨ B) -> (C -> D)) <-> (E ∨ F)
    tokens6 = tokenize_fol('((A ∨ B) -> (C -> D)) <-> (E ∨ F)')
    tokens6_after_elimination_implies = eliminate_implies(tokens6)
    tokens6_after_elimination_equivalence = eliminate_equivalence(tokens6_after_elimination_implies)
    print("\nTest Case 6:")
    print("Original Tokens:", tokens6)
    print("After Eliminating Implications:", tokens6_after_elimination_equivalence)
    nested_tokens6 = parse_nested(tokens6_after_elimination_equivalence)  # Convert to nested structure
    nnf6 = push_negations_inward(nested_tokens6)
    print("6. Nested Tokens:", nested_tokens6)
    print("6. NNF Result:", nnf6)
    simplify_nnf6 = simplify_parentheses(nnf6)
    print("6. Human read nnf:", tokenize_human_read_nnf(simplify_nnf6))
    print("6. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf6)))

    # Test Case 7: (¬A -> (B <-> ¬C)) ∧ (D -> (E -> F))
    tokens7 = tokenize_fol('(¬A -> (B <-> ¬C)) ∧ (D -> (E -> F))')
    tokens7_after_elimination_implies = eliminate_implies(tokens7)
    tokens7_after_elimination_equivalence = eliminate_equivalence(tokens7_after_elimination_implies)
    print("\nTest Case 7:")
    print("Original Tokens:", tokens7)
    print("After Eliminating Implications:", tokens7_after_elimination_equivalence)
    nested_tokens7 = parse_nested(tokens7_after_elimination_equivalence)  # Convert to nested structure
    nnf7 = push_negations_inward(nested_tokens7)
    print("7. Nested Tokens:", nested_tokens7)
    print("7. NNF Result:", nnf7)
    simplify_nnf7 = simplify_parentheses(nnf7)
    print("7. Human read nnf:", tokenize_human_read_nnf(simplify_nnf7))
    print("7. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_nnf7)))

    # Test Case 8: (C ∨ ¬D) -> (E ∧ F)
    tokens6 = tokenize_fol('(C ∨ ¬D) -> (E ∧ F)')
    tokens6_after_elimination_implies = eliminate_implies(tokens6)
    tokens6_after_elimination_equivalence = eliminate_equivalence(tokens6_after_elimination_implies)
    print("\nTest Case 8:")
    print("Original Tokens:", tokens6)
    print("After Eliminating Implications:", tokens6_after_elimination_equivalence)
    nested_tokens6 = parse_nested(tokens6_after_elimination_equivalence)  # Convert to nested structure
    nnf6 = push_negations_inward(nested_tokens6)
    print("8. Nested Tokens:", nested_tokens6)
    print("8. NNF Result:", nnf6)
    cnf6 = convert_to_cnf_inner(nnf6)
    print("8. CNF Result:", cnf6)
    simplify_cnf6 = simplify_parentheses(cnf6)
    print("8. Human read nnf:", tokenize_human_read_nnf(nnf6))
    print("8. Human read cnf:", tokenize_human_read_nnf(cnf6))
    print("8. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(nnf6)))
    print("8. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_cnf6)))

    # Test Case 9: (C ∨ ¬D) -> (E ∧ F)
    tokens6 = tokenize_fol('(¬man(x) ∨ mortal(x)) ∧ ¬man(a) -> mortal(a)')
    tokens6_after_elimination_implies = eliminate_implies(tokens6)
    tokens6_after_elimination_equivalence = eliminate_equivalence(tokens6_after_elimination_implies)
    print("\nTest Case 9:")
    print("Original Tokens:", tokens6)
    print("After Eliminating Implications:", tokens6_after_elimination_equivalence)
    nested_tokens6 = parse_nested(tokens6_after_elimination_equivalence)  # Convert to nested structure
    nnf6 = push_negations_inward(nested_tokens6)
    print("9. Nested Tokens:", nested_tokens6)
    print("9. NNF Result:", nnf6)
    cnf6 = convert_to_cnf_inner(nnf6)
    print("9. CNF Result:", cnf6)
    simplify_cnf6 = simplify_parentheses(cnf6)
    print("9. Human read nnf:", tokenize_human_read_nnf(nnf6))
    print("9. Human read cnf:", tokenize_human_read_nnf(cnf6))
    print("9. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(nnf6)))
    print("9. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(simplify_cnf6)))

    # (p(x) ∨ ¬q(y)) -> (p(y) ∧ q(x))

    # # Test Case 8: ((A <-> B) -> C) ∨ ((D ∧ E) <-> (F ∨ G))
    # tokens8 = tokenize_fol('((A <-> B) -> C) ∨ ((D ∧ E) <-> (F ∨ G))')
    # tokens8_after_elimination_implies = eliminate_implies(tokens8)
    # tokens8_after_elimination_equivalence = eliminate_equivalence(tokens8_after_elimination_implies)
    # print("\nTest Case 8:")
    # print("Original Tokens:", tokens8)
    # nested_tokens8 = parse_nested(tokens8_after_elimination_equivalence)  # Convert to nested structure
    # nnf8 = push_negations_inward(nested_tokens8)
    # print("8. Nested Tokens:", nested_tokens8)
    # print("8. NNF Result:", nnf8)
    # print("8. Human read nnf:", tokenize_human_read_nnf(nnf8))
    # print("8. Most simplified form", tokens_to_readable(tokenize_human_read_nnf(nnf8)))

    # # test 1 ¬ (A ∧ B)
    # tokens = ['¬', '(', 'A', '∧', 'B', ')']  # Flat token list
    # nested_tokens = parse_nested(tokens)  # Convert to nested structure
    # nnf = push_negations_inward(nested_tokens)
    # print("1. Nested Tokens:", nested_tokens)
    # print("1. NNF Result:", nnf)
    # print("1. Human read nnf:", tokenize_human_read_nnf(nnf))

    # # test 2 ¬ (¬ (A ∨ B) ∧ (B ∨ C))
    # tokens = ['¬', '(', '¬', '(', 'A', '∨', 'B', ')', '∧', '(', 'B', '∨', 'C', ')', ')']  # Input tokens
    # nested_tokens = parse_nested(tokens)  # Convert to nested structure
    # print("\n2. Nested Tokens:", nested_tokens)
    # nnf_result = push_negations_inward(nested_tokens)
    # print("2. NNF Result:", nnf_result)
    # print("2. Human read nnf:", tokenize_human_read_nnf(nnf_result))

    # # test 3 ¬((A ∧ (B ∨ C)) ∨ ¬(D ∧ E))
    # tokens = ['¬', '(', '(', 'A', '∧', '(', 'B', '∨', 'C', ')', ')', '∨', '¬', '(', 'D', '∧', 'E', ')', ')']
    # nested_tokens = parse_nested(tokens)
    # nnf_result = push_negations_inward(nested_tokens)
    # print("\n3. Nested Tokens:", nested_tokens)
    # print("3. NNF Result:", nnf_result)
    # print("3. Human read nnf:", tokenize_human_read_nnf(nnf_result))

    # # test 4 ¬ (happy(x) ∨ lucky(x))
    # tokens = ['¬', '(', 'happy(x)', '∨', 'lucky(y)', ')']
    # nested_tokens = parse_nested(tokens)
    # nnf_result = push_negations_inward(nested_tokens)
    # print("\n4. Nested Tokens:", nested_tokens)
    # print("4. NNF Result:", nnf_result)
    # print("4. Human read nnf:", tokenize_human_read_nnf(nnf_result))
