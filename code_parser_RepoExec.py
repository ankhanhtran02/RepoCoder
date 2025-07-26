def code_parser(response, target_func_prompt, function_signature):
    # remove redundant indentation
    def normalize_indentation(code_text):
        lines = code_text.split('\n')
        if not lines:
            return ""
            
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return ""
            
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
        
        normalized_lines = []
        for line in lines:
            if line.strip():
                normalized_lines.append(line[min_indent:])
            else:
                normalized_lines.append(line)
                
        return '\n'.join(normalized_lines)
    #Find python block.-.
    python_blocks = []
    start_idx = 0
    while True:
        start = response.find("```python", start_idx)
        if start == -1:
            break
        
        start += len("```python")
        end = response.find("```", start)
        
        if end == -1:
            normalized_code = normalize_indentation(response[start:].rstrip())
            python_blocks.append((start, len(response), normalized_code))
            break
        else:
            normalized_code = normalize_indentation(response[start:end].rstrip())
            python_blocks.append((start, end, normalized_code))
            start_idx = end + 3
    
    if python_blocks:
        parsed_code = None
        for i in range(len(python_blocks)):  # go upward
            candidate_code = python_blocks[i][2]
            if function_signature in candidate_code:
                parsed_code = candidate_code
                break

        # use the first block
        if parsed_code is None:
            parsed_code = python_blocks[0][2]
    else:
        parsed_code = response
    #  remove main if exists
    main_check = 'if __name__ == "__main__":'
    main_check_alt = "if __name__ == '__main__':" 
    
    if main_check in parsed_code:
        parsed_code = parsed_code[:parsed_code.find(main_check)].rstrip()
    elif main_check_alt in parsed_code:
        parsed_code = parsed_code[:parsed_code.find(main_check_alt)].rstrip()
    
    #  rremove import statement (import..., from...)
    lines = parsed_code.split('\n')
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line.startswith('import ') and not stripped_line.startswith('from '):
            filtered_lines.append(line)
    parsed_code = '\n'.join(filtered_lines)
    
    #   add target function prompt if not exists
    if not parsed_code.strip().startswith("from ") and not parsed_code.strip().startswith("import ") and not parsed_code.strip().startswith("def "):
        
        if (parsed_code.startswith("\n ") or 
            parsed_code.strip().startswith(" ") or 
            parsed_code.startswith("\t") or
            parsed_code.lstrip().startswith('"""') or 
            parsed_code.lstrip().startswith("'''") or
            parsed_code.lstrip().startswith('r"""') or
            parsed_code.lstrip().startswith("r'''")):
            lines = parsed_code.strip().split('\n')
            indented_lines = []
            is_raw_docstring = parsed_code.lstrip().startswith('r"""') or parsed_code.lstrip().startswith("r'''")
            for i, line in enumerate(lines):
                if line.strip():  
                    if i == 0 and is_raw_docstring:
                        if 'r"""' in line:
                            line = line.replace('r"""', '"""', 1)
                        elif "r'''" in line:
                            line = line.replace("r'''", "'''", 1)
                    if not line.startswith('    '):
                        indented_lines.append('    ' + line)
                    else:
                        indented_lines.append(line)
                else:
                    indented_lines.append(line)  
            indented_code = '\n'.join(indented_lines)
            result = target_func_prompt + indented_code
        else:
            result = parsed_code
    else:
        result = parsed_code
    if result.startswith(' ') or result.startswith('\t'):
        lines = result.split('\n')
        def_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                def_line_index = i
                break
        if def_line_index >= 0:
            def_indent = len(lines[def_line_index]) - len(lines[def_line_index].lstrip())
            if def_indent > 0:
                result = '\n'.join(
                    line[def_indent:] if line.strip() else line 
                    for line in lines
                )
        else:
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                if min_indent > 0:
                    result = '\n'.join(line[min_indent:] if line.strip() else line for line in lines)
    result = '\n'.join(line for line in result.split('\n') if line.strip())
    return result