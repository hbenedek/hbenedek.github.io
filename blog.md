---
layout: page
title: ""
---

Short posts on my side-projects:

<ul>
{% for post in site.posts %}
    {% if post.categories contains "blog" %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
{% endfor %}
</ul>