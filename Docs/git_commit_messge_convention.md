

## Git - Commit Message Convention
---
커밋 메시지를 작성할 때는 원칙을 정하고 일관성 있게 작성해야 한다. 아래는 유다시티의 커밋 메시지 스타일 가이드를 참조한 내용이다.

</br>

### 1. Commit Message Structure
---
기본적으로 커밋 메시지는 아래와 같이 제목/본문/꼬리말로 구성한다.
```
type : subject

body

footer
```

### 2. Commit Type
---
- Bug : 버그수정

- Feature : 새로운기능을추가하는경우

- Refactoring : 기존의코드구조를변경하는경우

- Doc : 문서를추가하거나수정하는경우

- Style : 코드의스타일을변경하는경우

- Test : Test 코드추가

### 3. Subject
---
- 제목은 50자를 넘기지 않고, 대문자로 작성하고 마침표를 붙이지 않는다.
- 과거시제를 사용하지 않고 명령어로 작성한다.
    - "Fixed" --> "Fix"
    - "Added" --> "Add"
    
### 4. Body
---
- 선택사항이기 때문에 모든 커밋에 분문내용을 작성할 필요는 없다.
- 부연설명이 필요하거나 커밋의 이유를 설명할 경우 작성해준다.
- 72자를 넘기지 않고 제목과 구분되기 위해 한칸을 띄워 작성한다.

### 5. Footer
---
- 선택사항이기 때문에 모든 커밋에 꼬리말을 작성할 필요는 없다.
- issue tracker id를 작성할 때 사용한다.
